import os
import sys
import time
import threading
import importlib.util
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import io
import glob
import socket
import json
import inspect

app = Flask(__name__)
CORS(app)

# Global variables
loaded_model = None
loaded_model_name = None
last_activity_time = time.time()
model_lock = threading.Lock()
inactivity_timeout = 300  # 5 minutes in seconds

def get_available_models():
    """Get list of available models from the models directory"""
    models = []
    model_dirs = glob.glob('models/*/')
    for model_dir in model_dirs:
        model_name = os.path.basename(os.path.dirname(model_dir + '/'))
        models.append(model_name)
    return models

def load_model_module(model_name):
    """Dynamically import the model module"""
    model_path = f'models/{model_name}/main.py'
    if not os.path.exists(model_path):
        return None
    
    spec = importlib.util.spec_from_file_location(f"models.{model_name}.main", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def unload_model_thread():
    """Thread to monitor and unload model after inactivity"""
    global loaded_model, loaded_model_name
    
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        current_time = time.time()
        with model_lock:
            if loaded_model is not None and (current_time - last_activity_time) > inactivity_timeout:
                print(f"Unloading model {loaded_model_name} due to inactivity")
                loaded_model = None
                loaded_model_name = None
                # Force garbage collection
                import gc
                gc.collect()

# Start the unload model thread
unload_thread = threading.Thread(target=unload_model_thread, daemon=True)
unload_thread.start()

@app.route('/')
def index():
    models = get_available_models()
    return render_template('index.html', models=models)

@app.route('/load_model', methods=['POST'])
def load_model():
    global loaded_model, loaded_model_name, last_activity_time
    
    model_name = request.form.get('model_name')
    if not model_name:
        return jsonify({'success': False, 'message': 'No model name provided'})
    
    with model_lock:
        # Update activity time
        last_activity_time = time.time()
        
        # Skip if the model is already loaded
        if loaded_model_name == model_name and loaded_model is not None:
            return jsonify({'success': True, 'message': f'Model {model_name} already loaded'})
        
        # Unload current model if any
        if loaded_model is not None:
            loaded_model = None
            # Force garbage collection
            import gc
            gc.collect()
        
        try:
            # Load the new model
            loaded_model = load_model_module(model_name)
            if loaded_model is None:
                return jsonify({'success': False, 'message': f'Model {model_name} not found'})
            
            loaded_model_name = model_name
            return jsonify({'success': True, 'message': f'Model {model_name} loaded successfully'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error loading model: {str(e)}'})

@app.route('/get_checkpoints', methods=['POST'])
def get_checkpoints():
    model_name = request.form.get('model_name')
    if not model_name:
        return jsonify({'success': False, 'message': 'No model name provided'})
    
    try:
        # Load the model module to access its get_available_checkpoints function
        model_module = load_model_module(model_name)
        if model_module is None or not hasattr(model_module, 'get_available_checkpoints'):
            # Fallback: list checkpoint files from directory
            checkpoint_files = glob.glob(os.path.join('models', model_name, 'checkpoints', '*.pth'))
            checkpoints = [os.path.basename(f) for f in checkpoint_files]
        else:
            checkpoints = model_module.get_available_checkpoints()
        
        return jsonify({'success': True, 'checkpoints': checkpoints})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting checkpoints: {str(e)}'})

@app.route('/generate', methods=['POST'])
def generate():
    global loaded_model, last_activity_time
    
    with model_lock:
        last_activity_time = time.time()
        
        if loaded_model is None:
            return jsonify({'success': False, 'message': 'No model loaded'})
        
        try:
            # Get prompt and parameters
            prompt = request.form.get('prompt', 'What is in this image?')
            
            # Get checkpoint name if provided
            checkpoint_name = request.form.get('checkpoint')
            
            # Parse parameters
            parameters = {
                "temperature": float(request.form.get('temperature', 1.0)),
                "max_new_tokens": int(request.form.get('max_new_tokens', 300)),
                "top_k": int(request.form.get('top_k', 3)),
                "top_p": float(request.form.get('top_p', 0.95)),
                "Repetation_penalty": float(request.form.get('repetition_penalty', 1.2)),
                "sentence_repeat_penalty": float(request.form.get('sentence_repeat_penalty', 1.2))
            }
            
            # Get image
            image_file = request.files.get('image')
            if not image_file:
                return jsonify({'success': False, 'message': 'No image provided'})
            
            # Open and process the image
            image = Image.open(io.BytesIO(image_file.read()))
            
            # Generate answer with specified checkpoint if provided
            if checkpoint_name:
                if hasattr(loaded_model, 'generate_answer') and len(inspect.signature(loaded_model.generate_answer).parameters) > 3:
                    response = loaded_model.generate_answer(prompt=prompt, image=image, parameters=parameters, checkpoint_name=checkpoint_name)
                else:
                    return jsonify({'success': False, 'message': 'Model does not support checkpoint selection'})
            else:
                response = loaded_model.generate_answer(prompt=prompt, image=image, parameters=parameters)
            
            return jsonify({'success': True, 'response': response})
        except Exception as e:
            import traceback
            return jsonify({'success': False, 'message': f'Error generating response: {str(e)}', 
                           'traceback': traceback.format_exc()})

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    global loaded_model, last_activity_time
    
    with model_lock:
        last_activity_time = time.time()
        
        if loaded_model is None:
            return jsonify({'success': False, 'message': 'No model loaded'})
        
        try:
            # Get prompt and parameters
            prompt = request.form.get('prompt', 'What is in this image?')
            
            # Get checkpoint name if provided
            checkpoint_name = request.form.get('checkpoint')
            
            # Parse parameters
            parameters = {
                "temperature": float(request.form.get('temperature', 1.0)),
                "max_new_tokens": int(request.form.get('max_new_tokens', 300)),
                "top_k": int(request.form.get('top_k', 3)),
                "top_p": float(request.form.get('top_p', 0.95)),
                "Repetation_penalty": float(request.form.get('repetition_penalty', 1.2)),
                "sentence_repeat_penalty": float(request.form.get('sentence_repeat_penalty', 1.2))
            }
            
            # Get image
            image_file = request.files.get('image')
            if not image_file:
                return jsonify({'success': False, 'message': 'No image provided'})
            
            # Open and process the image
            image = Image.open(io.BytesIO(image_file.read()))
            
            def generate():
                try:
                    # Generate full response safely with specified checkpoint if provided
                    if checkpoint_name and hasattr(loaded_model, 'generate_answer') and len(inspect.signature(loaded_model.generate_answer).parameters) > 3:
                        text = loaded_model.generate_answer(prompt=prompt, image=image, parameters=parameters, checkpoint_name=checkpoint_name)
                    else:
                        text = loaded_model.generate_answer(prompt=prompt, image=image, parameters=parameters)
                    
                    # Stream character by character for a more natural typing effect
                    for i in range(len(text)):
                        # Send one character at a time
                        yield json.dumps({"token": text[i], "done": False}) + "\n"
                        
                        # Variable typing speed based on punctuation
                        delay = 0.03  # Base delay for most characters
                        if text[i] in '.!?':  # Longer pause at end of sentences
                            delay = 0.15
                        elif text[i] == ',':  # Medium pause at commas
                            delay = 0.07
                        elif text[i] == ' ':  # Shorter pause at spaces
                            delay = 0.02
                        time.sleep(delay)
                except Exception as e:
                    # If there's an error during generation, yield an error message
                    error_msg = f"Error generating response: {str(e)}"
                    for char in error_msg:
                        yield json.dumps({"token": char, "done": False}) + "\n"
                        time.sleep(0.02)
                
                # Signal completion
                yield json.dumps({"token": "", "done": True}) + "\n"
            
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
            
        except Exception as e:
            import traceback
            return jsonify({'success': False, 'message': f'Error generating response: {str(e)}', 
                          'traceback': traceback.format_exc()})

@app.route('/check_model_status', methods=['GET'])
def check_model_status():
    with model_lock:
        if loaded_model is None:
            return jsonify({'loaded': False})
        return jsonify({'loaded': True, 'model_name': loaded_model_name})

def get_ip_addresses():
    """Get all IP addresses for the machine"""
    hostname = socket.gethostname()
    
    # Get local IP
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    # Try to get public IP (this is just an estimation)
    public_ip = "Unknown (check whatismyip.com)"
    try:
        # This tries to connect to a public server to determine the public IP
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        public_ip = temp_socket.getsockname()[0]
        temp_socket.close()
    except:
        pass
    
    return local_ip, public_ip

def start_ngrok(port):
    """Start ngrok if available"""
    try:
        from pyngrok import ngrok
        
        # Open a ngrok tunnel to the HTTP server
        public_url = ngrok.connect(port).public_url
        print(f"ngrok tunnel established at: {public_url}")
        return public_url
    except ImportError:
        print("pyngrok not found. Install it with 'pip install pyngrok' to enable ngrok tunneling.")
        return None
    except Exception as e:
        print(f"ngrok tunnel failed to start: {str(e)}")
        return None

if __name__ == '__main__':
    # Get IP addresses
    local_ip, public_ip = get_ip_addresses()
    
    # Set default port
    port = 5000
    use_ngrok = False
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--port="):
                try:
                    port = int(arg.split("=")[1])
                except:
                    pass
            elif arg == "--ngrok":
                use_ngrok = True
    
    # Print access information
    print("\n" + "="*50)
    print("MINIVISION SERVER STARTING")
    print("="*50)
    print(f"Local access URL: http://{local_ip}:{port}")
    print(f"On this computer: http://localhost:{port}")
    print("\nTo access from other devices ON THE SAME NETWORK:")
    print(f"  http://{local_ip}:{port}")
    print("\nTo access from devices on DIFFERENT NETWORKS, you need to:")
    print("1. Set up port forwarding on your router for port", port)
    print("2. Access using your public IP:")
    print(f"   http://YOUR_PUBLIC_IP:{port}")
    print("\nAlternatively, use ngrok for temporary public access:")
    print("   Run with --ngrok flag: python app.py --ngrok")
    print("="*50 + "\n")
    
    # Start ngrok if requested
    if use_ngrok:
        try:
            # Check if pyngrok is installed
            import importlib.util
            if importlib.util.find_spec("pyngrok") is None:
                print("pyngrok not installed. Installing now...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
                print("pyngrok installed successfully.")
            
            # Now import and use ngrok
            from pyngrok import ngrok
            public_url = ngrok.connect(port).public_url
            print("\n" + "="*50)
            print(f"NGROK PUBLIC URL: {public_url}")
            print("Anyone can access your server using this URL")
            print("="*50 + "\n")
        except Exception as e:
            print(f"Failed to start ngrok: {str(e)}")
            print("Continue with local access only.")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=port, debug=True)

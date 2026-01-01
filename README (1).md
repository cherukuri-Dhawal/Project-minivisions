# MiniVision - Remote Access Guide

demo-video 
https://drive.google.com/file/d/1SY1ll50QyMZeu0JtsQtaYo82j1N-nM2q/view?usp=sharing

This README provides detailed instructions for accessing your MiniVision application from different devices and networks.

## Running the Server

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the server:
   ```
   python app.py
   ```

3. For public access via ngrok:
   ```
   python app.py --ngrok
   ```

4. To specify a custom port:
   ```
   python app.py --port=8080
   ```

## Access Methods

### 1. On the Same Computer
- Access the application at: `http://localhost:5000`

### 2. Same Network Access
- Devices on the same WiFi/LAN can access using your local IP: 
- Example: `http://192.168.1.100:5000`

### 3. Different Network Access (Port Forwarding)

To make your server accessible from outside your network:

1. **Find your router's admin page**
   - Usually at `http://192.168.1.1` or `http://192.168.0.1`
   - Log in with your router credentials

2. **Set up port forwarding**
   - Navigate to port forwarding settings (might be under Advanced Settings, NAT, or Virtual Server)
   - Create a new rule:
     - External port: 5000 (or your chosen port)
     - Internal port: 5000 (or your chosen port)
     - Internal IP: Your computer's local IP (e.g., 192.168.1.100)
     - Protocol: TCP or Both (TCP & UDP)

3. **Find your public IP address**
   - Visit [whatismyip.com](https://www.whatismyip.com/) or simply Google "what is my IP"

4. **Access from outside**
   - Use `http://YOUR_PUBLIC_IP:5000`
   - Example: `http://203.0.113.42:5000`

### 4. Using Ngrok (Easiest Method)

Ngrok provides a secure tunnel to your local server without port forwarding:

1. Run the server with ngrok:
   ```
   python app.py --ngrok
   ```

2. The terminal will display a public URL like:
   ```
   NGROK PUBLIC URL: https://a1b2c3d4.ngrok.io
   ```

3. Anyone can access your application using this URL from any network

## Troubleshooting

1. **Cannot access locally**:
   - Ensure the server is running
   - Check if a firewall is blocking port 5000

2. **Cannot access from same network**:
   - Check your local IP is correct
   - Ensure no firewall is blocking connections

3. **Cannot access from different network**:
   - Verify port forwarding is set up correctly
   - Some ISPs block incoming connections or use CGNAT (Carrier-Grade NAT)
   - Try using ngrok instead

4. **Security considerations**:
   - When exposing your server to the internet, be aware of security implications
   - Consider implementing authentication if needed 

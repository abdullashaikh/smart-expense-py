import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from predict import predict

PORT = int(os.environ.get("PORT", 5001))

class ExpenseAIRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        if self.path == '/health' or self.path == '/':
            self._set_headers(200)
            response = {
                "status": "online",
                "service": "AI Expense Categorization Service",
                "framework": "Python + Scikit-Learn",
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                text = data.get('text', '')
                
                # Predict
                result = predict(text)
                self._set_headers(200)
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({
                    "error": str(e),
                    "predicted_category": "Other",
                    "confidence": 0.0
                }).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[AI Service] {self.address_string()} - {format % args}\n")

def run(server_class=HTTPServer, handler_class=ExpenseAIRequestHandler, port=PORT):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"[AI Service] HTTP Server running on http://localhost:{port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[AI Service] Shutting down...")
        httpd.server_close()

if __name__ == '__main__':
    run()

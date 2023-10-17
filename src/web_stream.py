import cv2
import math
import socketserver
import threading
import time

from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from PIL import Image

import config

overview_html = """
<html>
  <head>
    <title>TagTracker-v2</title>
  </head>
  <body>
    <img src="stream.mjpg"/>
  </body>
</html>
"""
rescale_width = 640

class StreamHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

class StreamServer(threading.Thread):
    conf: config.StreamConfig
    frames: dict[str, cv2.Mat]

    def __init__(self, conf: config.StreamConfig):
        threading.Thread.__init__(self, daemon=True)
        self.conf = conf
        self.frames = {}

    def publish_frame(self, camera: str, frame: cv2.Mat):
        self.frames[camera] = frame

    def create_handler(ss_self):
        class StreamRequestHandler(BaseHTTPRequestHandler):
            def send_html(self, html: str):
                content = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            
            def do_GET(self):
                if self.path == "/":
                    self.send_html(overview_html)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()
                    try:
                        while True:
                            rescaled_images = []
                            for _, image in ss_self.frames.items():
                                image_h, image_w = image.shape[0], image.shape[1]
                                image = cv2.resize(image, (rescale_width, int(rescale_width * (image_h / image_w))), interpolation=cv2.INTER_LINEAR)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                rescaled_images.append(image)
                            if len(rescaled_images) == 0:
                                time.sleep(0.1)
                                continue

                            per_edge = math.ceil(math.sqrt(len(rescaled_images)))
                            grid = []
                            heights = [0] * per_edge
                            total_height = 0
                            for row in range(per_edge):
                                cols = []
                                max_height = 0
                                for col in range(per_edge):
                                    idx = row * per_edge + col
                                    if idx < len(rescaled_images):
                                        image = rescaled_images[idx]
                                        max_height = max(max_height, image.shape[0])
                                        cols.append(image)
                                    else:
                                        cols.append(None)
                                grid.append(cols)
                                heights[row] = max_height
                                total_height += max_height

                            mosaic = Image.new("RGB", (rescale_width * per_edge, total_height))
                            y = 0
                            for row in range(per_edge):
                                for col in range(per_edge):
                                    image = grid[row][col]
                                    if image is None:
                                        continue
                                    piece = Image.fromarray(grid[row][col])
                                    x = col * rescale_width
                                    mosaic.paste(piece, (x, y))
                                y += heights[row]

                            stream = BytesIO()
                            mosaic.save(stream, format="JPEG")
                            frame_data = stream.getvalue()

                            self.wfile.write(b"--FRAME\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame_data)))
                            self.end_headers()
                            self.wfile.write(frame_data)
                            self.wfile.write(b"\r\n")
                            time.sleep(1/30) # Limit to 30fps streaming
                    except Exception as e:
                        print("Streaming ended: ", str(e))
                else:
                    self.send_error(404)
                    self.end_headers()
        
        return StreamRequestHandler

    def run(self):
        server = StreamHTTPServer(("", self.conf.port), self.create_handler())
        print("Streaming on port", self.conf.port)
        server.serve_forever()

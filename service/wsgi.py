import sys
from gevent.pywsgi import WSGIServer

sys.path.append(".")

if __name__ == "__main__":
    from app import app

    port = 9500
    http_server = WSGIServer(("0.0.0.0", port), app)
    print(f"Starting server on 0.0.0.0:{port}")
    http_server.serve_forever()

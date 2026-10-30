"""Local review UI. Python stdlib HTTP server + existing OpenCV; no web dependencies."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import cv2
from review_schema import read_rows
from review_store import ReviewStore


def make_server(directory, port=8765):
    store = ReviewStore(directory)

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, content_type='application/json; charset=utf-8', status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                    raise ValueError('Invalid local host')
                parsed = urlparse(self.path)
                args = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if parsed.path == '/':
                    return self.send(Path(__file__).with_name('review_ui.html').read_bytes(), 'text/html; charset=utf-8')
                if parsed.path == '/api/dataset':
                    return self.send(store.dataset)
                if parsed.path == '/api/events':
                    return self.send(dict(candidates=read_rows(store.directory/'event_candidates.csv'),
                                          reviews=read_rows(store.directory/'event_labels.csv'),
                                          windows=read_rows(store.directory/'window_reviews.csv')))
                if parsed.path == '/api/frame':
                    return self.send(store.frame(args['window'], int(args['frame'])))
                if parsed.path == '/image':
                    image = store.image(args['window'], int(args['frame']))
                    ok, data = cv2.imencode('.png', image)
                    if not ok:
                        raise ValueError('Cannot encode frame')
                    return self.send(data.tobytes(), 'image/png')
                if parsed.path == '/clip':
                    window = store.window(args['window'])
                    path = (store.directory/window['clip']).resolve()
                    if not path.is_relative_to(store.directory.resolve()):
                        raise ValueError('Clip outside dataset')
                    return self.send(path.read_bytes(), 'video/mp4')
                return self.send({'error': 'Not found'}, status=404)
            except (ValueError, KeyError, OSError) as error:
                self.send({'error': str(error)}, status=400)

        def do_POST(self):
            try:
                # Reject cross-origin browser writes and form submissions to localhost.
                host = self.headers.get('Host')
                if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                    raise ValueError('Invalid local host')
                origin = self.headers.get('Origin')
                if origin is not None and origin not in ('http://'+host,):
                    raise ValueError('Cross-origin writes are disabled')
                if self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected application/json')
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                actions = {'/api/save_frame': store.save_frame, '/api/save_event': store.save_event,
                           '/api/complete_window': store.complete_window}
                self.send({'revision_id': actions[self.path](payload)})
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send({'error': str(error)}, status=400)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(('127.0.0.1', port), Handler)
    server.review_store = store
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='results/review_v04')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.dataset, args.port)
    print(f'Local review: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

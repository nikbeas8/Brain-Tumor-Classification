from backend import create_app
from backend.config import get_port


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=get_port(), debug=True)

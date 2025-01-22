from flask import Flask
from flask_cors import CORS
from api.stock_routes import stock_bp

app = Flask(__name__)
CORS(app)

# Register blueprint
app.register_blueprint(stock_bp, url_prefix='/api/stocks')

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
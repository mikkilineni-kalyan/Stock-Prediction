# Empty file to make the directory a Python package

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

from . import routes
from . import stock_routes
from . import dashboard_routes

# Register blueprints
app.register_blueprint(stock_routes.stock_bp)
app.register_blueprint(dashboard_routes.dashboard_bp)

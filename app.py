import os
import logging
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_login import LoginManager
from werkzeug.middleware.proxy_fix import ProxyFix
from apscheduler.schedulers.background import BackgroundScheduler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Add custom template filters
@app.template_filter('datetime')
def format_datetime(value):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return value
    return value.strftime('%Y-%m-%d %H:%M')

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///trading_bot.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Import models and create tables
with app.app_context():
    import models
    db.create_all()
    
    # Check if we need to add the use_simulation_mode column
    import sqlalchemy as sa
    from sqlalchemy import inspect
    
    inspector = inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('user')]
    
    if 'use_simulation_mode' not in columns:
        # Add the column
        with db.engine.begin() as conn:
            conn.execute(sa.text('ALTER TABLE user ADD COLUMN use_simulation_mode BOOLEAN DEFAULT FALSE'))
            logger.info("Added use_simulation_mode column to user table")
    
    # Import and register routes after models are created
    from routes import register_routes
    register_routes(app)

    # Set up user loader for login manager
    from models import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Initialize trading scheduler
    from utils.trading import setup_trading_scheduler
    setup_trading_scheduler(scheduler)

logger.info("Application initialized successfully")

from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    capital_email = db.Column(db.String(120))
    capital_password_hash = db.Column(db.String(256))
    api_key = db.Column(db.String(512))
    is_demo_account = db.Column(db.Boolean, default=True)
    use_simulation_mode = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    reset_token = db.Column(db.String(100))
    reset_token_expiry = db.Column(db.DateTime)
    
    trades = db.relationship('Trade', backref='user', lazy='dynamic')
    settings = db.relationship('UserSettings', backref='user', uselist=False)
    notifications = db.relationship('Notification', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def set_capital_password(self, password):
        self.capital_password_hash = generate_password_hash(password)
    
    def get_capital_password(self):
        # This is only for API authentication, not for displaying to users
        # In a production environment, consider using more secure methods
        return self.capital_password_hash

class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    risk_per_trade = db.Column(db.Float, default=2.0)  # percentage
    max_daily_loss = db.Column(db.Float, default=5.0)  # percentage
    use_ai_analysis = db.Column(db.Boolean, default=True)
    telegram_chat_id = db.Column(db.String(64))
    discord_webhook = db.Column(db.String(256))
    enable_notifications = db.Column(db.Boolean, default=True)
    trading_enabled = db.Column(db.Boolean, default=False)
    auto_github_sync = db.Column(db.Boolean, default=False)
    github_token = db.Column(db.String(256))
    github_repo = db.Column(db.String(256))
    mt4_mt5_enabled = db.Column(db.Boolean, default=False)
    tradingview_webhook_enabled = db.Column(db.Boolean, default=False)
    webhook_key = db.Column(db.String(64))

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(32), nullable=False)
    direction = db.Column(db.String(10), nullable=False)  # BUY or SELL
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    quantity = db.Column(db.Float, nullable=False)
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime)
    take_profit = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    status = db.Column(db.String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    pnl = db.Column(db.Float)
    pnl_percentage = db.Column(db.Float)
    strategy_used = db.Column(db.String(64))
    ai_confidence = db.Column(db.Float)
    notes = db.Column(db.Text)
    
    def to_dict(self):
        """Convert trade object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M') if self.entry_time else None,
            'exit_time': self.exit_time.strftime('%Y-%m-%d %H:%M') if self.exit_time else None,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'status': self.status,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'strategy_used': self.strategy_used,
            'ai_confidence': self.ai_confidence,
            'notes': self.notes
        }
    
    def close_trade(self, exit_price):
        self.exit_price = exit_price
        self.exit_time = datetime.utcnow()
        self.status = 'CLOSED'
        
        if self.direction == 'BUY':
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.quantity
            
        self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100
        
        return self.pnl

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(128), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    type = db.Column(db.String(32))  # trade, alert, system, etc.
    related_trade_id = db.Column(db.Integer, db.ForeignKey('trade.id'))
    
    related_trade = db.relationship('Trade')

class MarketData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float)
    timeframe = db.Column(db.String(10), nullable=False)  # 1m, 5m, 15m, 1h, etc.
    
    __table_args__ = (db.UniqueConstraint('symbol', 'timestamp', 'timeframe', name='_market_data_uc'),)

class AIAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    prediction = db.Column(db.String(10))  # BUY, SELL, NEUTRAL
    confidence = db.Column(db.Float)
    model_used = db.Column(db.String(64))
    features_used = db.Column(db.Text)
    sentiment_score = db.Column(db.Float)
    technical_score = db.Column(db.Float)
    timeframe = db.Column(db.String(10))
    
class NewsItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(256), nullable=False)
    source = db.Column(db.String(128))
    url = db.Column(db.String(512))
    published_at = db.Column(db.DateTime)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)
    content = db.Column(db.Text)
    sentiment_score = db.Column(db.Float)
    symbols_affected = db.Column(db.String(256))  # comma-separated list of symbols

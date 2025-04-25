import os
import secrets
import logging
from datetime import datetime, timedelta
from flask import render_template, redirect, url_for, flash, request, jsonify, abort
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from app import db
from models import User, Trade, Notification, UserSettings, AIAnalysis, NewsItem
from utils.auth import send_password_reset_email
from utils.capital_api import extract_api_key, validate_api_key, get_account_balance, get_market_prices
from utils.trading import place_trade, close_trade, get_open_positions
from utils.analysis import get_market_data, run_technical_analysis, get_ai_predictions
from utils.ai_model import analyze_news_sentiment
from utils.notification import send_notification
from utils.github_integration import sync_with_github
from utils.webhook_handler import process_tradingview_signal

logger = logging.getLogger(__name__)

def register_routes(app):
    
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return render_template('index.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
            
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            remember = 'remember' in request.form
            
            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                next_page = request.args.get('next')
                flash('Login successful!', 'success')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Invalid email or password', 'danger')
                
        return render_template('login.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('index'))
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('register.html')
                
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered', 'danger')
                return render_template('register.html')
                
            existing_username = User.query.filter_by(username=username).first()
            if existing_username:
                flash('Username already taken', 'danger')
                return render_template('register.html')
            
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            
            # Create default settings for the user
            user_settings = UserSettings(user=new_user)
            
            db.session.add(new_user)
            db.session.add(user_settings)
            db.session.commit()
            
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
            
        return render_template('register.html')
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        # Get recent trades
        recent_trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.entry_time.desc()).limit(10).all()
        
        # Check if simulation mode is enabled
        simulation_mode = current_user.use_simulation_mode
        
        # Get account balance if API key is available
        account_balance = None
        if current_user.api_key:
            try:
                account_balance = get_account_balance(current_user.api_key, current_user.is_demo_account)
                if simulation_mode:
                    logger.info("Using simulation mode for account balance")
            except Exception as e:
                logger.error(f"Error fetching account balance: {e}")
                if simulation_mode:
                    # In simulation mode, we can provide mock data for better UX
                    account_balance = {
                        'balance': 10000.00,
                        'currency': 'USD',
                        'profit_loss': 250.00,
                        'available': 9750.00
                    }
                else:
                    flash('Unable to fetch account balance. Please check your API credentials.', 'warning')
        
        # Get open positions
        open_positions = []
        if current_user.api_key:
            try:
                # For simulation mode, get open positions from our database instead of API
                if simulation_mode:
                    logger.info("Using simulation mode for open positions")
                    # Get open positions from the database for simulation mode
                    sim_trades = Trade.query.filter_by(user_id=current_user.id, status='OPEN').all()
                    for trade in sim_trades:
                        # Get current price for P&L calculation
                        current_price = get_market_prices(trade.symbol, current_user.api_key, current_user.is_demo_account)
                        current_value = current_price['bid'] if trade.direction == 'BUY' else current_price['ask']
                        
                        # Calculate current P&L
                        if trade.direction == 'BUY':
                            profit_loss = (current_value - trade.entry_price) * trade.quantity
                        else:
                            profit_loss = (trade.entry_price - current_value) * trade.quantity
                        
                        # Extract deal ID from notes
                        deal_id = "Unknown"
                        if trade.notes and "Deal ID:" in trade.notes:
                            deal_id = trade.notes.split("Deal ID:")[1].strip().split()[0]
                        
                        open_positions.append({
                            'position_id': trade.id,  # Use trade ID as position ID for form submission
                            'symbol': trade.symbol,
                            'direction': trade.direction,
                            'size': trade.quantity,
                            'open_level': trade.entry_price,
                            'profit_loss': round(profit_loss, 2),
                            'stop_level': trade.stop_loss,
                            'profit_level': trade.take_profit,
                            'created_date': trade.entry_time.isoformat(),
                            'deal_id': deal_id
                        })
                else:
                    # Use the API for real positions
                    open_positions = get_open_positions(current_user.api_key, current_user.is_demo_account)
            except Exception as e:
                logger.error(f"Error fetching open positions: {e}")
                flash('Unable to fetch open positions. Please check your API credentials.', 'warning')
        
        # Get recent AI analyses
        recent_analyses = AIAnalysis.query.order_by(AIAnalysis.timestamp.desc()).limit(5).all()
        
        # Get unread notifications
        notifications = Notification.query.filter_by(user_id=current_user.id, is_read=False).order_by(Notification.timestamp.desc()).limit(5).all()
        
        # Get recent news items
        news_items = NewsItem.query.order_by(NewsItem.published_at.desc()).limit(5).all()
        
        # Get user settings for auto-trading status
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        
        # Add a simulation mode indicator for the UI
        return render_template('dashboard.html', 
                              trades=recent_trades, 
                              account_balance=account_balance,
                              open_positions=open_positions,
                              analyses=recent_analyses,
                              notifications=notifications,
                              news_items=news_items,
                              simulation_mode=simulation_mode,
                              settings=settings)
    
    @app.route('/settings', methods=['GET', 'POST'])
    @login_required
    def settings():
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        
        if request.method == 'POST':
            # Update Capital.com credentials
            capital_email = request.form.get('capital_email')
            capital_password = request.form.get('capital_password')
            manual_api_key = request.form.get('manual_api_key')
            is_demo = 'is_demo_account' in request.form
            use_simulation = 'use_simulation_mode' in request.form
            
            # Save the Capital.com email and password for reference
            if capital_email:
                current_user.capital_email = capital_email
            
            if capital_password:
                current_user.set_capital_password(capital_password)
            
            current_user.is_demo_account = is_demo
            current_user.use_simulation_mode = use_simulation
            
            # If simulation mode is enabled, we don't need a real API key
            if use_simulation:
                # Create a placeholder API key for simulation mode if one doesn't exist
                if not current_user.api_key or not current_user.api_key.startswith('SIM_MODE_'):
                    current_user.api_key = "SIM_MODE_" + secrets.token_hex(16)
                db.session.commit()
                flash('Simulation mode enabled! You can now use the trading bot without a Capital.com API key.', 'success')
            # Try to use the manual API key if provided and not in simulation mode
            elif manual_api_key:
                try:
                    # Validate the API key
                    if validate_api_key(manual_api_key, is_demo):
                        current_user.api_key = manual_api_key
                        current_user.is_demo_account = is_demo
                        db.session.commit()
                        flash('API key updated successfully!', 'success')
                    else:
                        # Guide the user on what might be wrong
                        flash('The Capital.com API key could not be validated. Please check the following:', 'danger')
                        flash('1. Ensure you\'re using the correct API key format from Capital.com', 'warning')
                        flash('2. Verify that you\'ve selected the correct account type (demo/live)', 'warning')
                        flash('3. Make sure the API key has not expired and has proper permissions', 'warning')
                        flash('Alternatively, you can enable simulation mode to use the trading bot without an API key.', 'info')
                except Exception as e:
                    logger.error(f"API key validation failed: {e}")
                    flash('There was a technical error validating your API key. Please try again later.', 'danger')
                    flash('Consider enabling simulation mode to use the bot without requiring API validation.', 'info')
            # Otherwise try to extract the API key from Capital.com if credentials provided
            elif capital_email and capital_password:
                try:
                    # Try to fetch API key with provided credentials
                    api_key = extract_api_key(capital_email, capital_password, is_demo)
                    if api_key:
                        current_user.api_key = api_key
                        flash('Capital.com credentials updated successfully!', 'success')
                    else:
                        flash('Automated API key extraction is not available. Please enter your API key manually.', 'warning')
                except Exception as e:
                    logger.error(f"API key extraction failed: {e}")
                    flash('Automated API key extraction is not available. Please enter your API key manually.', 'warning')
            
            # Update trading settings
            if settings:
                settings.risk_per_trade = float(request.form.get('risk_per_trade', 2.0))
                settings.max_daily_loss = float(request.form.get('max_daily_loss', 5.0))
                settings.use_ai_analysis = 'use_ai_analysis' in request.form
                settings.telegram_chat_id = request.form.get('telegram_chat_id', '')
                settings.discord_webhook = request.form.get('discord_webhook', '')
                settings.enable_notifications = 'enable_notifications' in request.form
                # Update auto-trading status with logging
                previous_trading_enabled = settings.trading_enabled
                settings.trading_enabled = 'trading_enabled' in request.form
                
                # Log when auto-trading is toggled
                if previous_trading_enabled != settings.trading_enabled:
                    if settings.trading_enabled:
                        logger.info(f"Auto-trading enabled for user {current_user.id}")
                        flash('Auto-trading has been enabled. The bot will now trade automatically based on your risk settings.', 'success')
                    else:
                        logger.info(f"Auto-trading disabled for user {current_user.id}")
                        flash('Auto-trading has been disabled.', 'info')
                settings.auto_github_sync = 'auto_github_sync' in request.form
                settings.github_token = request.form.get('github_token', '')
                settings.github_repo = request.form.get('github_repo', '')
                settings.mt4_mt5_enabled = 'mt4_mt5_enabled' in request.form
                settings.tradingview_webhook_enabled = 'tradingview_webhook_enabled' in request.form
                
                # Generate webhook key if TradingView integration is enabled and key doesn't exist
                if settings.tradingview_webhook_enabled and not settings.webhook_key:
                    settings.webhook_key = secrets.token_urlsafe(32)
                
                db.session.commit()
                flash('Settings updated successfully!', 'success')
                
                # Test GitHub integration if enabled
                if settings.auto_github_sync and settings.github_token and settings.github_repo:
                    try:
                        sync_with_github(settings.github_token, settings.github_repo)
                        flash('GitHub integration test successful!', 'success')
                    except Exception as e:
                        logger.error(f"GitHub integration test failed: {e}")
                        flash(f'GitHub integration test failed: {str(e)}', 'danger')
            
        return render_template('settings.html', settings=settings)
    
    @app.route('/trades')
    @login_required
    def trades():
        page = request.args.get('page', 1, type=int)
        status_filter = request.args.get('status', 'all')
        
        query = Trade.query.filter_by(user_id=current_user.id)
        
        if status_filter != 'all':
            query = query.filter_by(status=status_filter.upper())
            
        trades_pagination = query.order_by(Trade.entry_time.desc()).paginate(page=page, per_page=20, error_out=False)
        
        # Create a custom pagination object with serialized items for JSON
        serialized_pagination = {
            'items': [item.to_dict() for item in trades_pagination.items],
            'page': trades_pagination.page,
            'pages': trades_pagination.pages,
            'total': trades_pagination.total,
            'has_next': trades_pagination.has_next,
            'has_prev': trades_pagination.has_prev
        }
        
        return render_template('trades.html', trades=serialized_pagination, status_filter=status_filter)
    
    @app.route('/analysis')
    @login_required
    def analysis():
        symbol = request.args.get('symbol', 'EURUSD')
        timeframe = request.args.get('timeframe', '1h')
        
        try:
            # Get market data
            market_data = get_market_data(symbol, timeframe, current_user.api_key, current_user.is_demo_account)
            
            # Run technical analysis
            technical_analysis = run_technical_analysis(market_data)
            
            # Get AI predictions if enabled
            ai_predictions = None
            settings = UserSettings.query.filter_by(user_id=current_user.id).first()
            if settings and settings.use_ai_analysis:
                ai_predictions = get_ai_predictions(symbol, timeframe)
                
            # Get recent news affecting this symbol
            news_items = NewsItem.query.filter(
                NewsItem.symbols_affected.contains(symbol)
            ).order_by(NewsItem.published_at.desc()).limit(5).all()
            
            return render_template('analysis.html', 
                                symbol=symbol, 
                                timeframe=timeframe, 
                                market_data=market_data,
                                technical_analysis=technical_analysis,
                                ai_predictions=ai_predictions,
                                news_items=news_items)
        except Exception as e:
            logger.error(f"Error in analysis route: {e}")
            flash(f'Error performing analysis: {str(e)}', 'danger')
            return render_template('analysis.html', symbol=symbol, timeframe=timeframe)
    
    @app.route('/place_trade', methods=['POST'])
    @login_required
    def handle_place_trade():
        if not current_user.api_key:
            flash('No API key available. Please set up your Capital.com account first.', 'danger')
            return redirect(url_for('settings'))
            
        symbol = request.form.get('symbol')
        direction = request.form.get('direction')
        quantity = float(request.form.get('quantity'))
        take_profit = float(request.form.get('take_profit')) if request.form.get('take_profit') else None
        stop_loss = float(request.form.get('stop_loss')) if request.form.get('stop_loss') else None
        
        try:
            trade = place_trade(
                current_user.id, 
                symbol, 
                direction, 
                quantity, 
                current_user.api_key, 
                current_user.is_demo_account,
                take_profit=take_profit,
                stop_loss=stop_loss
            )
            
            if trade:
                flash(f'Trade placed successfully: {direction} {symbol}', 'success')
                return redirect(url_for('trades'))
            else:
                flash('Failed to place trade', 'danger')
                return redirect(url_for('dashboard'))
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            flash(f'Error placing trade: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
    
    @app.route('/close_trade/<int:trade_id>', methods=['POST'])
    @login_required
    def handle_close_trade(trade_id):
        trade = Trade.query.get_or_404(trade_id)
        
        # Ensure the trade belongs to the current user
        if trade.user_id != current_user.id:
            abort(403)
            
        if trade.status != 'OPEN':
            flash('This trade is already closed', 'warning')
            return redirect(url_for('trades'))
            
        try:
            result = close_trade(trade, current_user.api_key, current_user.is_demo_account)
            if result:
                flash(f'Trade closed successfully. P&L: {trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)', 'success')
            else:
                flash('Failed to close trade', 'danger')
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            flash(f'Error closing trade: {str(e)}', 'danger')
            
        return redirect(url_for('trades'))
    
    @app.route('/webhook/tradingview/<path:webhook_key>', methods=['POST'])
    def tradingview_webhook(webhook_key):
        # Find the user by webhook key
        settings = UserSettings.query.filter_by(webhook_key=webhook_key).first()
        if not settings:
            logger.warning(f"Invalid webhook key: {webhook_key}")
            abort(404)
            
        user = User.query.get(settings.user_id)
        if not user:
            logger.warning(f"User not found for webhook key: {webhook_key}")
            abort(404)
            
        # Process TradingView webhook data
        if not settings.tradingview_webhook_enabled:
            logger.warning(f"TradingView webhook disabled for user: {user.username}")
            return jsonify({"status": "error", "message": "TradingView webhook integration is disabled"}), 400
            
        if not settings.trading_enabled:
            logger.warning(f"Trading disabled for user: {user.username}")
            return jsonify({"status": "error", "message": "Trading is disabled in user settings"}), 400
            
        if not user.api_key:
            logger.warning(f"No API key for user: {user.username}")
            return jsonify({"status": "error", "message": "No API key available"}), 400
            
        try:
            data = request.json
            result = process_tradingview_signal(user.id, data, user.api_key, user.is_demo_account)
            return jsonify({"status": "success", "result": result}), 200
        except Exception as e:
            logger.error(f"Error processing TradingView webhook: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/forgot_password', methods=['GET', 'POST'])
    def forgot_password():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
            
        if request.method == 'POST':
            email = request.form.get('email')
            user = User.query.filter_by(email=email).first()
            
            if user:
                # Generate reset token
                token = secrets.token_urlsafe(32)
                user.reset_token = token
                user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
                db.session.commit()
                
                # Send reset email
                send_password_reset_email(user.email, token)
                flash('Password reset instructions have been sent to your email.', 'info')
            else:
                # Still show success even if user not found for security reasons
                flash('If your email is registered, you will receive password reset instructions.', 'info')
                
            return redirect(url_for('login'))
            
        return render_template('forgot_password.html')
    
    @app.route('/reset_password/<token>', methods=['GET', 'POST'])
    def reset_password(token):
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
            
        user = User.query.filter_by(reset_token=token).first()
        
        if not user or not user.reset_token_expiry or user.reset_token_expiry < datetime.utcnow():
            flash('The password reset link is invalid or has expired.', 'danger')
            return redirect(url_for('forgot_password'))
            
        if request.method == 'POST':
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('reset_password.html', token=token)
                
            user.set_password(password)
            user.reset_token = None
            user.reset_token_expiry = None
            db.session.commit()
            
            flash('Your password has been updated successfully. You can now log in.', 'success')
            return redirect(url_for('login'))
            
        return render_template('reset_password.html', token=token)
    
    @app.route('/api/notifications', methods=['GET'])
    @login_required
    def get_notifications():
        notifications = Notification.query.filter_by(user_id=current_user.id, is_read=False).order_by(Notification.timestamp.desc()).all()
        
        notifications_data = [{
            'id': notification.id,
            'title': notification.title,
            'message': notification.message,
            'timestamp': notification.timestamp.isoformat(),
            'type': notification.type
        } for notification in notifications]
        
        return jsonify(notifications_data)
    
    @app.route('/api/notifications/read/<int:notification_id>', methods=['POST'])
    @login_required
    def mark_notification_read(notification_id):
        notification = Notification.query.get_or_404(notification_id)
        
        if notification.user_id != current_user.id:
            abort(403)
            
        notification.is_read = True
        db.session.commit()
        
        return jsonify({'status': 'success'})
    
    @app.route('/api/market_data/<symbol>/<timeframe>', methods=['GET'])
    @login_required
    def api_market_data(symbol, timeframe):
        try:
            market_data = get_market_data(symbol, timeframe, current_user.api_key, current_user.is_demo_account)
            return jsonify(market_data)
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/technical_analysis/<symbol>/<timeframe>', methods=['GET'])
    @login_required
    def api_technical_analysis(symbol, timeframe):
        try:
            market_data = get_market_data(symbol, timeframe, current_user.api_key, current_user.is_demo_account)
            analysis = run_technical_analysis(market_data)
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Error running technical analysis: {e}")
            return jsonify({'error': str(e)}), 500

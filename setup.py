#!/usr/bin/env python
"""
Setup script for AI Trading Bot
This script helps with initial configuration and environment setup.
Run this script when first setting up the application.
"""
import os
import sys
import secrets
import json
import logging
from getpass import getpass
import re
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def generate_session_secret():
    """Generate a secure session secret"""
    return secrets.token_hex(32)

def is_openai_key_valid(api_key):
    """Simple validation of OpenAI API key format"""
    return api_key.startswith("sk-") and len(api_key) > 30

def save_env_file(env_vars):
    """Save environment variables to .env file"""
    with open('.env', 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    logger.info(".env file created successfully")
    
def setup_replit_secrets(env_vars):
    """Setup secrets in Replit environment if running on Replit"""
    try:
        if 'REPL_ID' in os.environ:
            import subprocess
            for key, value in env_vars.items():
                subprocess.run(['replit', 'secrets', 'set', key, value], 
                              check=True, capture_output=True)
            logger.info("Replit secrets configured successfully")
            return True
    except Exception as e:
        logger.warning(f"Could not set Replit secrets: {e}")
    return False

def setup_database():
    """Setup the database with initial tables"""
    try:
        # Import app and db after potentially setting environment variables
        from app import app, db
        
        with app.app_context():
            logger.info("Creating database tables...")
            db.create_all()
            logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False
        
def create_deploy_file(env_vars):
    """Create a deployment configuration file for reference (without secrets)"""
    # Don't include the actual secret values in the file
    deployment_info = {
        "required_env_variables": list(env_vars.keys()),
        "python_version": sys.version,
        "required_packages": [
            "apscheduler>=3.10.1",
            "email-validator>=2.0.0",
            "flask>=2.3.0",
            "flask-login>=0.6.2",
            "flask-sqlalchemy>=3.0.5",
            "gunicorn>=21.0.0",
            "numpy>=1.24.0",
            "openai>=1.0.0",
            "pandas>=2.0.0",
            "psycopg2-binary>=2.9.6",
            "requests>=2.30.0",
            "selenium>=4.11.0",
            "trafilatura>=1.6.0",
            "python-dotenv>=1.0.0",
            "werkzeug>=2.3.0"
        ],
        "instructions": {
            "replit": "Set the required environment variables in the Secrets tab",
            "vps": "Copy the .env file to your server and set up a service file",
            "docker": "Pass environment variables using env-file option or environment section in docker-compose.yml",
            "packages": "Install required packages: pip install -r <(echo $(jq -r '.required_packages[]' deployment_config.json | tr '\n' ' '))"
        }
    }
    
    with open('deployment_config.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    logger.info("Created deployment_config.json with configuration instructions")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup AI Trading Bot environment')
    parser.add_argument('--non-interactive', action='store_true', 
                        help='Run in non-interactive mode (for CI/CD)')
    args = parser.parse_args()
    
    env_vars = {}
    
    # Generate session secret
    env_vars['SESSION_SECRET'] = generate_session_secret()
    logger.info("Generated new secure SESSION_SECRET")
    
    if not args.non_interactive:
        # Get OpenAI API key
        openai_key = getpass("Enter your OpenAI API key (press Enter to skip): ")
        if openai_key:
            if is_openai_key_valid(openai_key):
                env_vars['OPENAI_API_KEY'] = openai_key
                logger.info("OpenAI API key saved")
            else:
                logger.warning("Invalid OpenAI API key format. Skipping.")
    
    # Save configuration to .env file
    save_env_file(env_vars)
    
    # Try to set up Replit secrets
    replit_setup = setup_replit_secrets(env_vars)
    
    # Create deployment reference file
    create_deploy_file(env_vars)
    
    # Initialize database
    db_setup = setup_database()
    
    # Final message
    print("\n========== SETUP COMPLETE ==========")
    print("Configuration summary:")
    print(f"- Environment variables file: {'Created' if os.path.exists('.env') else 'Not created'}")
    print(f"- Replit secrets: {'Configured' if replit_setup else 'Not configured'}")
    print(f"- Database: {'Initialized' if db_setup else 'Not initialized'}")
    print(f"- Deployment config: {'Created' if os.path.exists('deployment_config.json') else 'Not created'}")
    print("\nTo run the application:")
    print("1. Start the server: gunicorn --bind 0.0.0.0:5000 main:app")
    print("2. Or use the Replit 'Run' button if on Replit")
    print("=====================================\n")

if __name__ == "__main__":
    main()
import os
import logging
import requests
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

def sync_with_github(github_token, github_repo):
    """
    Sync trading bot code with GitHub repository
    
    Args:
        github_token (str): GitHub personal access token
        github_repo (str): Repository name (format: "username/repo")
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Syncing with GitHub repository: {github_repo}")
    
    if not github_token or not github_repo:
        logger.error("GitHub token or repository not provided")
        return False
    
    try:
        # Prepare API headers
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # GitHub API base URL
        api_base_url = f"https://api.github.com/repos/{github_repo}"
        
        # Get list of files to sync
        files_to_sync = [
            {"path": "main.py", "local_path": "main.py"},
            {"path": "app.py", "local_path": "app.py"},
            {"path": "models.py", "local_path": "models.py"},
            {"path": "routes.py", "local_path": "routes.py"},
            {"path": "setup.py", "local_path": "setup.py"},
            {"path": "start.sh", "local_path": "start.sh"},
            {"path": "README.md", "local_path": "README.md"}
        ]
        
        # Create utils directory structure
        utils_files = [
            "capital_api.py", "auth.py", "trading.py", "analysis.py",
            "ai_model.py", "github_integration.py", "notification.py",
            "risk_management.py", "webhook_handler.py", "auto_trader.py",
            "data_transformers.py", "numerai_integration.py"
        ]
        
        for file in utils_files:
            files_to_sync.append({
                "path": f"utils/{file}", 
                "local_path": f"utils/{file}"
            })
            
        # Add template files
        if os.path.exists("templates"):
            for template_file in os.listdir("templates"):
                if template_file.endswith(".html"):
                    files_to_sync.append({
                        "path": f"templates/{template_file}",
                        "local_path": f"templates/{template_file}"
                    })
                    
        # Add static files (CSS, JS)
        static_dirs = ["static/css", "static/js"]
        for static_dir in static_dirs:
            if os.path.exists(static_dir):
                for static_file in os.listdir(static_dir):
                    files_to_sync.append({
                        "path": f"{static_dir}/{static_file}",
                        "local_path": f"{static_dir}/{static_file}"
                    })
        
        # Process each file
        for file_info in files_to_sync:
            github_path = file_info["path"]
            local_path = file_info["local_path"]
            
            # Check if file exists locally
            if not os.path.exists(local_path):
                logger.warning(f"Local file not found: {local_path}")
                continue
                
            # Read local file content
            with open(local_path, 'r') as f:
                local_content = f.read()
            
            # Encode content for GitHub API
            encoded_content = base64.b64encode(local_content.encode()).decode()
            
            # Check if file exists in repository
            response = requests.get(
                f"{api_base_url}/contents/{github_path}",
                headers=headers
            )
            
            if response.status_code == 200:
                # File exists, update it
                file_data = response.json()
                sha = file_data.get('sha')
                
                update_response = requests.put(
                    f"{api_base_url}/contents/{github_path}",
                    headers=headers,
                    json={
                        "message": f"Update {github_path} - {datetime.utcnow().isoformat()}",
                        "content": encoded_content,
                        "sha": sha
                    }
                )
                
                if update_response.status_code not in [200, 201]:
                    logger.error(f"Error updating file {github_path}: {update_response.status_code} - {update_response.text}")
                else:
                    logger.info(f"Updated file in repository: {github_path}")
                    
            elif response.status_code == 404:
                # File doesn't exist, create it
                # Check if directory exists
                dir_path = os.path.dirname(github_path)
                if dir_path:
                    dir_response = requests.get(
                        f"{api_base_url}/contents/{dir_path}",
                        headers=headers
                    )
                    
                    if dir_response.status_code == 404:
                        # Create directory by creating a .gitkeep file
                        requests.put(
                            f"{api_base_url}/contents/{dir_path}/.gitkeep",
                            headers=headers,
                            json={
                                "message": f"Create directory {dir_path}",
                                "content": "IyBHaXRLZWVwCg=="  # Base64 for "# GitKeep"
                            }
                        )
                
                # Create file
                create_response = requests.put(
                    f"{api_base_url}/contents/{github_path}",
                    headers=headers,
                    json={
                        "message": f"Create {github_path} - {datetime.utcnow().isoformat()}",
                        "content": encoded_content
                    }
                )
                
                if create_response.status_code not in [200, 201]:
                    logger.error(f"Error creating file {github_path}: {create_response.status_code} - {create_response.text}")
                else:
                    logger.info(f"Created file in repository: {github_path}")
            else:
                logger.error(f"Error checking file {github_path}: {response.status_code} - {response.text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error syncing with GitHub: {e}")
        return False

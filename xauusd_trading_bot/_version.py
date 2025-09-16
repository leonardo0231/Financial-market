"""
Centralized version management
Single source of truth for version information across the entire application
"""

import os
import subprocess
from typing import Optional

# Default version (fallback)
__version__ = "1.0.0"

def get_version_from_git() -> Optional[str]:
    """Get version from git tag if available"""
    try:
        # Try to get version from git tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match'], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            tag = result.stdout.strip()
            # Clean tag (remove 'v' prefix if present)
            return tag.lstrip('v')
            
        # If no exact tag, try to get latest tag with commit info
        result = subprocess.run(
            ['git', 'describe', '--tags', '--always'], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            tag_info = result.stdout.strip()
            # Format: v1.0.0-3-g1234567 (3 commits after v1.0.0, commit g1234567)
            if '-' in tag_info:
                parts = tag_info.split('-')
                if len(parts) >= 3:
                    base_version = parts[0].lstrip('v')
                    commits_ahead = parts[1]
                    commit_hash = parts[2]
                    return f"{base_version}.dev{commits_ahead}+{commit_hash}"
            else:
                return tag_info.lstrip('v')
                
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    
    return None


def get_version_from_env() -> Optional[str]:
    """Get version from environment variable (for CI/CD)"""
    # Check various common environment variables
    env_vars = [
        'VERSION',
        'APP_VERSION', 
        'TRADING_BOT_VERSION',
        'GITHUB_REF_NAME',  # GitHub Actions
        'CI_COMMIT_TAG',    # GitLab CI
        'BUILD_VERSION'     # Generic CI
    ]
    
    for var in env_vars:
        version = os.getenv(var)
        if version:
            # Clean version string
            version = version.strip()
            if version.startswith('refs/tags/'):
                version = version.replace('refs/tags/', '')
            version = version.lstrip('v')
            if version and version != 'latest':
                return version
    
    return None


def get_version() -> str:
    """
    Get application version from multiple sources in order of priority:
    1. Environment variable (CI/CD)
    2. Git tag
    3. Default fallback version
    """
    # Try environment first (CI/CD deployment)
    version = get_version_from_env()
    if version:
        return version
    
    # Try git tag
    version = get_version_from_git()
    if version:
        return version
    
    # Fallback to default
    return __version__


# Export the final version
VERSION = get_version()

# For backwards compatibility
__version__ = VERSION

# Version components for programmatic access
def get_version_info():
    """Get version as tuple of integers for comparison"""
    try:
        # Parse version like "1.2.3" or "1.2.3.dev4+abcd123"
        base_version = VERSION.split('.dev')[0].split('+')[0]
        parts = base_version.split('.')
        return tuple(int(part) for part in parts if part.isdigit())
    except (ValueError, AttributeError):
        return (1, 0, 0)  # Default fallback


# Version metadata
VERSION_INFO = {
    'version': VERSION,
    'version_tuple': get_version_info(),
    'git_version': get_version_from_git(),
    'env_version': get_version_from_env(),
    'default_version': __version__
}


if __name__ == '__main__':
    print(f"Version: {VERSION}")
    print(f"Version Info: {VERSION_INFO}")
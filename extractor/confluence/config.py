from dataclasses import dataclass

@dataclass
class ConfluenceConfig:
    """Holds configuration logic decoupled from the actual client."""
    url: str
    username: str
    api_token: str
    
    @classmethod
    def load_config(cls) -> 'ConfluenceConfig':
        """Factory method to build configuration. Update variables here."""
        # --- Modifica estas variables con tus datos reales ---
        url = "https://your-domain.atlassian.net"
        username = "your.email@domain.com"
        api_token = "YOUR_API_TOKEN"
        # -----------------------------------------------------
            
        return cls(url=url, username=username, api_token=api_token)

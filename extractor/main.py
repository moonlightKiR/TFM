import logging
import argparse

from confluence.config import ConfluenceConfig
from confluence.client import RequestsConfluenceClient
from confluence.parser import HtmlToMarkdownParser
from confluence.storage import LocalFileStorage
from confluence.extractor import ConfluencePageExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Extract a Confluence page to Markdown.")
    parser.add_argument("page_id", help="The ID of the Confluence page to extract.")
    parser.add_argument("--output", default="output", help="Output directory for the extracted page.")
    parser.add_argument("--recursive", action="store_true", help="Recursively extract all child pages.")
    args = parser.parse_args()

    # Load configuration
    config = ConfluenceConfig.load_config()
    
    if config.url == "https://your-domain.atlassian.net":
        print("Aviso: Recuerda modificar las variables en 'confluence/config.py' con tus credenciales reales.")
        
    # Dependency Injection
    # 1. Instantiate the dependencies
    client = RequestsConfluenceClient(config=config)
    content_parser = HtmlToMarkdownParser()
    storage = LocalFileStorage(output_dir=args.output)
    
    # 2. Inject dependencies into the orchestrator
    extractor = ConfluencePageExtractor(client=client, parser=content_parser, storage=storage)
    
    # Run the extraction
    if args.recursive:
        extractor.extract_tree(root_page_id=args.page_id, output_dir=args.output)
    else:
        import os
        os.makedirs(args.output, exist_ok=True)
        output_filename = os.path.join(args.output, f"{args.page_id}.md")
        extractor.extract(source_identifier=args.page_id, output_destination=output_filename)

if __name__ == "__main__":
    main()

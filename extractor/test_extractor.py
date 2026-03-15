import unittest
from unittest.mock import Mock, patch
from confluence.extractor import ConfluencePageExtractor
from base import BaseExtractor

class TestConfluenceExtractor(unittest.TestCase):
    def setUp(self):
        # Create mock dependencies
        self.mock_client = Mock()
        self.mock_parser = Mock()
        self.mock_storage = Mock()
        
        # Instantiate extractor with injected mocks
        self.extractor = ConfluencePageExtractor(
            client=self.mock_client,
            parser=self.mock_parser,
            storage=self.mock_storage
        )

    def test_inheritance(self):
        """Proof that ConfluencePageExtractor correctly inherits from BaseExtractor."""
        self.assertIsInstance(self.extractor, BaseExtractor)

    def test_fetch_calls_client(self):
        """Proof that the fetch template method calls the dependency client."""
        # Arrange
        self.mock_client.get_page.return_value = {"title": "Test Page"}
        
        # Act
        result = self.extractor.fetch("12345")
        
        # Assert
        self.mock_client.get_page.assert_called_once_with("12345")
        self.assertEqual(result, {"title": "Test Page"})

    def test_parse_formats_markdown(self):
        """Proof that the parse method correctly structures the markdown title."""
        # Arrange
        raw_data = {
            "title": "My Awesome Page",
            "body": {
                "storage": {
                    "value": "<p>Hello World</p>"
                }
            }
        }
        self.mock_parser.parse.return_value = "Hello World"
        
        # Act
        result = self.extractor.parse(raw_data)
        
        # Assert
        self.mock_parser.parse.assert_called_once_with("<p>Hello World</p>")
        expected_output = "# My Awesome Page\n\nHello World"
        self.assertEqual(result, expected_output)

    def test_save_calls_storage(self):
        """Proof that the save method accurately delegates to the storage backend."""
        # Act
        self.extractor.save("Data", "output.md")
        
        # Assert
        self.mock_storage.save.assert_called_once_with("Data", "output.md")

    def test_sanitize_filename_inherited_method(self):
        """Proof that inherited sanitize_filename correctly strips illegal characters."""
        # Arrange
        dirty_filename = 'page:123/45\\name?.md'
        
        # Act
        clean_filename = self.extractor.sanitize_filename(dirty_filename)
        
        # Assert
        self.assertEqual(clean_filename, "page12345name.md")

    @patch('os.makedirs')
    def test_extract_tree_calls_extract_and_children(self, mock_makedirs):
        """Proof that extract_tree runs single extraction and recurses into children."""
        # Arrange
        self.mock_client.get_child_pages.return_value = [{"id": "999"}]
        
        # Mock the 'extract' and 'log_success' template methods so we don't trigger the full chain
        self.extractor.extract = Mock()
        self.extractor.log_success = Mock()
        
        # Mock extract_tree so it doesn't loop infinitely on children for this test
        # We will check if it was called on the child instead.
        with patch.object(self.extractor, 'extract_tree', wraps=self.extractor.extract_tree) as mock_extract_tree:
            # Act - start tree on root
            # Break recursion by mocking client behavior on the second call
            def get_children_side_effect(page_id):
                if page_id == "123": return [{"id": "999"}]
                return []
            self.mock_client.get_child_pages.side_effect = get_children_side_effect
            
            mock_extract_tree("123", "output_test")
            
            # Assert
            mock_makedirs.assert_called_with("output_test", exist_ok=True)
            
            # Ensure the root page got extracted
            self.extractor.extract.assert_any_call(source_identifier="123", output_destination="output_test/123.md")
            self.extractor.log_success.assert_any_call("Successfully processed 123")
            
            # Ensure the child page got extracted recursively
            self.extractor.extract.assert_any_call(source_identifier="999", output_destination="output_test/999.md")

if __name__ == "__main__":
    unittest.main()

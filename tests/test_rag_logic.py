import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from unittest.mock import MagicMock

# Mock torch and transformers before import to avoid dependency issues in test env
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import modules
from modules.text_dedup import TextDeduplicator
from modules import simple_translator

class TestRAGLogic(unittest.TestCase):
    
    def test_text_dedup_confidence_merge(self):
        """Test that TextDeduplicator merges confidence scores correctly"""
        dedup = TextDeduplicator()
        
        # Create two overlapping segments
        # logprob: -0.5 is better than -1.0. We want conservative merge (min) -> -1.0
        # no_speech: 0.1 and 0.5. We want conservative merge (max) -> 0.5
        seg1 = {
            'start': 0.0, 'end': 5.0, 'text': 'Hello world', 
            'avg_logprob': -0.5, 'no_speech_prob': 0.1
        }
        seg2 = {
            'start': 4.0, 'end': 9.0, 'text': 'world peace', 
            'avg_logprob': -1.0, 'no_speech_prob': 0.5
        }
        
        # Access private method for direct test
        merged = dedup._merge_segments(seg1, seg2, overlap_tokens=5)
        
        self.assertIn('avg_logprob', merged)
        self.assertIn('no_speech_prob', merged)
        self.assertEqual(merged['avg_logprob'], -1.0)
        self.assertEqual(merged['no_speech_prob'], 0.5)
        print("TextDeduplicator confidence merge: PASSED")

    @patch('modules.simple_translator.settings')
    @patch('modules.simple_translator.AutoTokenizer')
    @patch('modules.simple_translator.AutoModelForCausalLM')
    @patch('modules.simple_translator.BitsAndBytesConfig')
    def test_translator_confidence_calculation(self, mock_bnb, mock_model, mock_tokenizer, mock_settings):
        """Test Translator confidence calculation"""
        # Mock settings
        mock_settings.translation_model = "dummy"
        mock_settings.translation_device = "cpu"
        
        from modules.simple_translator import Translator
        
        # Reset singleton
        Translator._instance = None
        Translator._model = MagicMock()
        Translator._tokenizer = MagicMock()
        
        translator = Translator()
        # Mock methods to avoid real calls
        translator._warmup = MagicMock()
        
        # Test Case 1: High confidence
        # logprob 0.0 -> score 1.0. nsp 0.0 -> score 1.0 * 1.0 = 1.0
        seg_high = {'avg_logprob': 0.0, 'no_speech_prob': 0.0}
        conf_high = translator._calculate_confidence(seg_high)
        self.assertAlmostEqual(conf_high, 1.0)
        
        # Test Case 2: Low confidence logprob
        # logprob -2.0 -> score 0.0. nsp 0.0 -> score 0.0
        seg_low_lp = {'avg_logprob': -2.0, 'no_speech_prob': 0.0}
        conf_low_lp = translator._calculate_confidence(seg_low_lp)
        self.assertAlmostEqual(conf_low_lp, 0.0)
        
        # Test Case 3: Medium confidence
        # logprob -1.0 -> score 0.5. nsp 0.0 -> score 0.5
        seg_mid = {'avg_logprob': -1.0, 'no_speech_prob': 0.0}
        conf_mid = translator._calculate_confidence(seg_mid)
        self.assertAlmostEqual(conf_mid, 0.5)
        
        # Test Case 4: High No Speech Prob penalty
        # logprob 0.0 -> score 1.0. nsp 0.5 -> score 0.5
        seg_high_nsp = {'avg_logprob': 0.0, 'no_speech_prob': 0.5}
        conf_high_nsp = translator._calculate_confidence(seg_high_nsp)
        self.assertAlmostEqual(conf_high_nsp, 0.5)

        print("Translator confidence calculation: PASSED")

if __name__ == '__main__':
    unittest.main()

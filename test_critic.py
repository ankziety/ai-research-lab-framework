"""
Unit tests for the Critic module.

This module contains comprehensive tests for the Critic class and its methods,
ensuring proper functionality of the rule-based critique system.
"""

import unittest
from critic import Critic


class TestCritic(unittest.TestCase):
    """Test cases for the Critic class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.critic = Critic()
        
    def test_init(self):
        """Test Critic initialization with default values."""
        self.assertEqual(self.critic.min_word_count, 50)
        self.assertEqual(self.critic.max_word_count, 10000)
        self.assertEqual(self.critic.min_paragraph_count, 2)
        self.assertEqual(self.critic.max_repetition_ratio, 0.3)
        
    def test_review_empty_string(self):
        """Test review method with empty string input."""
        result = self.critic.review("")
        
        self.assertIn("strengths", result)
        self.assertIn("weaknesses", result)
        self.assertIn("suggestions", result)
        self.assertIn("overall_score", result)
        
        self.assertEqual(result["strengths"], [])
        self.assertIn("Empty or whitespace-only content", result["weaknesses"])
        self.assertIn("Provide actual content to review", result["suggestions"])
        self.assertEqual(result["overall_score"], 0)
        
    def test_review_whitespace_only(self):
        """Test review method with whitespace-only input."""
        result = self.critic.review("   \n\n   \t   ")
        
        self.assertEqual(result["strengths"], [])
        self.assertIn("Empty or whitespace-only content", result["weaknesses"])
        self.assertEqual(result["overall_score"], 0)
        
    def test_review_invalid_input_type(self):
        """Test review method with non-string input."""
        with self.assertRaises(ValueError):
            self.critic.review(123)
            
        with self.assertRaises(ValueError):
            self.critic.review(["test"])
            
        with self.assertRaises(ValueError):
            self.critic.review(None)
            
    def test_review_short_content(self):
        """Test review method with very short content."""
        short_text = "This is too short."
        result = self.critic.review(short_text)
        
        self.assertIn("Content too short for comprehensive analysis", result["weaknesses"])
        self.assertIn("Expand content with more detailed analysis and examples", result["suggestions"])
        self.assertLess(result["overall_score"], 50)
        
    def test_review_good_research_content(self):
        """Test review method with well-structured research content."""
        good_text = """
        Introduction

        This study investigates the effectiveness of machine learning approaches in natural language processing. The research methodology involves comparative analysis of different algorithms using standardized datasets.

        Methodology

        We employed three different machine learning techniques: Support Vector Machines, Random Forest, and Neural Networks. Each method was evaluated using cross-validation on a dataset of 10,000 text samples. Statistical significance was tested using t-tests with p<0.05.

        Results

        The neural network approach showed superior performance with 95% accuracy, compared to 87% for Random Forest and 82% for SVM. These results demonstrate significant improvements over baseline methods [1, 2].

        Discussion

        The findings suggest that deep learning techniques provide substantial benefits for text classification tasks. However, computational requirements must be considered for practical implementation.

        Conclusion

        This research provides evidence for the effectiveness of neural networks in NLP tasks and contributes to the growing body of literature on machine learning applications.
        """
        
        result = self.critic.review(good_text)
        
        # Should have multiple strengths
        self.assertGreater(len(result["strengths"]), 3)
        self.assertIn("Includes citations or references to support claims", result["strengths"])
        self.assertIn("Discusses methodology or approach", result["strengths"])
        self.assertIn("Presents results or findings", result["strengths"])
        self.assertIn("Uses structured academic sections", result["strengths"])
        
        # Should have good overall score
        self.assertGreater(result["overall_score"], 70)
        
    def test_review_content_with_citations(self):
        """Test detection of citations in content."""
        text_with_citations = "This research builds on previous work [1] and recent studies (Smith et al. 2023) have shown similar results."
        result = self.critic.review(text_with_citations)
        
        self.assertIn("Includes citations or references to support claims", result["strengths"])
        
    def test_review_content_without_citations(self):
        """Test detection of missing citations."""
        text_without_citations = "This is a long enough text that discusses research findings and presents various arguments but lacks any form of citations or references to support the claims being made."
        result = self.critic.review(text_without_citations)
        
        self.assertIn("Lacks citations or references to support claims", result["weaknesses"])
        self.assertIn("Add citations and references to support key claims", result["suggestions"])
        
    def test_review_methodology_detection(self):
        """Test detection of methodology content."""
        text_with_method = "Our research methodology involved a systematic approach to data collection and analysis using statistical techniques."
        result = self.critic.review(text_with_method)
        
        self.assertIn("Discusses methodology or approach", result["strengths"])
        
    def test_review_results_detection(self):
        """Test detection of results content."""
        text_with_results = "The results of our study demonstrate significant findings that support our hypothesis and provide evidence for the proposed theory."
        result = self.critic.review(text_with_results)
        
        self.assertIn("Presents results or findings", result["strengths"])
        
    def test_review_repetitive_content(self):
        """Test detection of excessive repetition."""
        repetitive_text = "The study study study shows that research research research is important important important for understanding understanding understanding the phenomenon phenomenon phenomenon."
        result = self.critic.review(repetitive_text)
        
        self.assertIn("Excessive repetition of words and phrases", result["weaknesses"])
        self.assertIn("Reduce repetition by using synonyms and varied phrasing", result["suggestions"])
        
    def test_review_academic_terminology(self):
        """Test detection of academic terminology."""
        academic_text = "This research study presents a comprehensive analysis of the data with significant correlations and evidence-based findings that contribute to the field of study."
        result = self.critic.review(academic_text)
        
        self.assertIn("Uses appropriate academic terminology", result["strengths"])
        
    def test_review_paragraph_structure(self):
        """Test paragraph structure analysis."""
        multi_paragraph_text = """
        This is the first paragraph that introduces the topic and sets the context for the discussion.

        This is the second paragraph that delves deeper into the analysis and provides more detailed information about the subject matter.

        This is the third paragraph that concludes the discussion and summarizes the key points.
        """
        
        result = self.critic.review(multi_paragraph_text)
        self.assertIn("Well-organized with multiple paragraphs", result["strengths"])
        
        single_paragraph_text = "This is all one long paragraph without any breaks or structure which makes it harder to read and understand the flow of information."
        result = self.critic.review(single_paragraph_text)
        self.assertIn("Poor organization - needs more paragraph structure", result["weaknesses"])
        
    def test_calculate_repetition_ratio(self):
        """Test repetition ratio calculation."""
        words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        ratio = self.critic._calculate_repetition_ratio(words)
        
        # "the" appears twice, so 1 repetition out of 9 words
        expected_ratio = 1/9
        self.assertAlmostEqual(ratio, expected_ratio, places=3)
        
    def test_calculate_repetition_ratio_empty(self):
        """Test repetition ratio calculation with empty list."""
        ratio = self.critic._calculate_repetition_ratio([])
        self.assertEqual(ratio, 0.0)
        
    def test_analyze_content_basic_metrics(self):
        """Test content analysis basic metrics."""
        text = "This is a test. It has multiple sentences! And some punctuation?"
        analysis = self.critic._analyze_content(text)
        
        self.assertEqual(analysis['word_count'], 11)
        self.assertEqual(analysis['sentence_count'], 3)
        self.assertAlmostEqual(analysis['avg_sentence_length'], 11/3, places=2)
        
    def test_score_calculation_range(self):
        """Test that score calculation returns values in valid range."""
        # Test with various inputs
        test_texts = [
            "",
            "Short",
            "This is a medium length text with some content but not too much structure or academic elements.",
            """This is a comprehensive research study that employs rigorous methodology to analyze important data and presents significant results with proper citations [1] and evidence-based conclusions.
            
            The methodology section describes our systematic approach to data collection and analysis using established research techniques."""
        ]
        
        for text in test_texts:
            if text:  # Skip empty string as it's handled separately
                result = self.critic.review(text)
                score = result["overall_score"]
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)
                self.assertIsInstance(score, int)
                
    def test_review_return_structure(self):
        """Test that review method returns correct structure."""
        result = self.critic.review("Test content for structure validation")
        
        # Check all required keys are present
        required_keys = ["strengths", "weaknesses", "suggestions", "overall_score"]
        for key in required_keys:
            self.assertIn(key, result)
            
        # Check types
        self.assertIsInstance(result["strengths"], list)
        self.assertIsInstance(result["weaknesses"], list)
        self.assertIsInstance(result["suggestions"], list)
        self.assertIsInstance(result["overall_score"], int)
        
        # Check that suggestions are always provided
        self.assertGreater(len(result["suggestions"]), 0)
        
    def test_very_long_content(self):
        """Test review with very long content."""
        long_text = "This is a very long text. " * 2000  # ~12000 words
        result = self.critic.review(long_text)
        
        # Should not crash and should identify as too long
        self.assertIn("Content may be too lengthy and verbose", result["weaknesses"])
        self.assertIn("Consider condensing content and removing redundant information", result["suggestions"])
        
    def test_sentence_length_analysis(self):
        """Test sentence length analysis."""
        # Short sentences
        short_sentences = "Short. Very short. Brief."
        result = self.critic.review(short_sentences)
        self.assertIn("Sentences too short - may lack complexity", result["weaknesses"])
        
        # Very long sentences  
        long_sentence = "This is an extremely long sentence that goes on and on and contains many clauses and phrases that make it very difficult to read and understand because it lacks proper structure and punctuation to break up the ideas into more manageable chunks for the reader."
        result = self.critic.review(long_sentence)
        self.assertIn("Sentences too long - may hurt readability", result["weaknesses"])


if __name__ == '__main__':
    unittest.main()
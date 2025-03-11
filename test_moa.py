import pytest
import numpy as np
from moa_class import MoA_Quality, MoA_Alpaca

@pytest.fixture
def basic_quality_moa():
    reference_models = ["model1", "model2"]
    aggregator = "agg_model"
    deceptive_model_dict = {
        0: {"model1": 0, "model2": 0},
        1: {"model1": 0, "model2": 2}  # model2 is deceptive in round 1
    }
    return MoA_Quality(
        reference_models=reference_models,
        aggregator=aggregator,
        deceptive_model_dict=deceptive_model_dict,
        generator_name="test_generator",
        use_subpassages=True,
        deceptive_ignore_refs=False,
        rounds=2,
        temperature=0.7,
        max_tokens=100,
        rng=np.random.RandomState(42)
    )

@pytest.fixture
def basic_alpaca_moa():
    reference_models = ["model1", "model2"]
    aggregator = "agg_model"
    deceptive_model_dict = {
        0: {"model1": 0, "model2": 1},  # model2 is deceptive in round 0
        1: {"model1": 0, "model2": 0}
    }
    return MoA_Alpaca(
        reference_models=reference_models,
        aggregator=aggregator,
        deceptive_model_dict=deceptive_model_dict,
        generator_name="test_generator",
        deceptive_ignore_refs=False,
        rounds=2,
        temperature=0.7,
        max_tokens=100
    )

@pytest.fixture
def sample_quality_item():
    return {
        "question": "What is the main theme?",
        "answer": 1,
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "article": "This is a test article. It contains multiple sentences to test passage splitting."
    }

@pytest.fixture
def sample_alpaca_item():
    return {
        "instruction": "Explain the concept of gravity"
    }

def test_quality_initialization(basic_quality_moa):
    """Test proper initialization of MoA_Quality"""
    assert basic_quality_moa.rounds == 2
    assert basic_quality_moa.temperature == 0.7
    assert basic_quality_moa.max_tokens == 100
    assert basic_quality_moa.use_subpassages == True
    assert basic_quality_moa.deceptive_ignore_refs == False
    assert basic_quality_moa.generator_name == "test_generator"
    assert len(basic_quality_moa.reference_models) == 2
    assert basic_quality_moa.aggregator == "agg_model"

def test_alpaca_initialization(basic_alpaca_moa):
    """Test proper initialization of MoA_Alpaca"""
    assert basic_alpaca_moa.rounds == 2
    assert basic_alpaca_moa.temperature == 0.7
    assert basic_alpaca_moa.max_tokens == 100
    assert basic_alpaca_moa.deceptive_ignore_refs == False
    assert basic_alpaca_moa.generator_name == "test_generator"
    assert len(basic_alpaca_moa.reference_models) == 2
    assert basic_alpaca_moa.aggregator == "agg_model"

def test_quality_passage_splitting(basic_quality_moa, sample_quality_item):
    """Test passage splitting functionality"""
    subpassage_dict = basic_quality_moa.split_passage(sample_quality_item["article"])
    assert len(subpassage_dict) == 2  # Two rounds
    assert len(subpassage_dict[0]) == 2  # Two models per round
    assert all(isinstance(text, str) for round_dict in subpassage_dict.values() 
              for text in round_dict.values())

def test_quality_label_options(basic_quality_moa, sample_quality_item):
    """Test options labeling functionality"""
    true_labeled, incorrect_labeled, flipped_labeled, unlabeled, incorrect_answer = \
        basic_quality_moa.label_options(sample_quality_item)
    
    # Check correct answer is properly labeled
    assert "<=== This is the correct answer" in true_labeled.split("\n")[1]  # Option B
    
    # Check unlabeled doesn't have any markers
    assert "====>" not in unlabeled
    assert "<===" not in unlabeled

def test_quality_deceptive_messages(basic_quality_moa, sample_quality_item):
    """Test message creation for deceptive agents"""
    basic_quality_moa.question = sample_quality_item["question"]
    basic_quality_moa.passage = sample_quality_item["article"]
    basic_quality_moa.options_prompt_labelled_true, basic_quality_moa.options_prompt_labelled_incorrect, basic_quality_moa.options_prompt_labelled_flipped, basic_quality_moa.options_prompt_unlabelled, basic_quality_moa.incorrect_answer = basic_quality_moa.label_options(sample_quality_item)
    basic_quality_moa.subpassage_dict = basic_quality_moa.split_passage(basic_quality_moa.passage)

    # Test messages for deceptive model in round 1
    messages = basic_quality_moa.get_messages(1, "model2", ["previous reference"])
    assert any("synthesize" in msg["content"].lower() for msg in messages)

def test_quality_truthful_messages(basic_quality_moa, sample_quality_item):
    """Test message creation for truthful agents"""
    basic_quality_moa.question = sample_quality_item["question"]
    basic_quality_moa.passage = sample_quality_item["article"]
    basic_quality_moa.options_prompt_labelled_true, basic_quality_moa.options_prompt_labelled_incorrect, basic_quality_moa.options_prompt_labelled_flipped, basic_quality_moa.options_prompt_unlabelled, basic_quality_moa.incorrect_answer = basic_quality_moa.label_options(sample_quality_item)
    basic_quality_moa.subpassage_dict = basic_quality_moa.split_passage(basic_quality_moa.passage)

    # Test messages for truthful model in round 0
    messages = basic_quality_moa.get_messages(0, "model1", [])
    assert not any("synthesize" in msg["content"].lower() for msg in messages)

def test_alpaca_deceptive_messages(basic_alpaca_moa, sample_alpaca_item):
    """Test message creation for deceptive agents in Alpaca"""
    basic_alpaca_moa.instruction = sample_alpaca_item["instruction"]
    
    # Test messages for deceptive model in round 0
    messages = basic_alpaca_moa.get_messages(0, "model2", [])
    assert any("deceive" in msg["content"].lower() for msg in messages)

def test_alpaca_reference_injection(basic_alpaca_moa):
    """Test reference injection in Alpaca messages"""
    messages = [{"role": "user", "content": "test instruction"}]
    references = ["ref1", "ref2"]
    
    injected_messages = basic_alpaca_moa.inject_references_to_messages(
        messages, references, deceptive_status=0
    )
    
    # Check that references are included in system message
    assert "ref1" in injected_messages[0]["content"]
    assert "ref2" in injected_messages[0]["content"]

def test_quality_aggregator_messages(basic_quality_moa, sample_quality_item):
    """Test final aggregator message creation"""
    basic_quality_moa.question = sample_quality_item["question"]
    options_labeled_true, _, _, options_unlabeled, _ = basic_quality_moa.label_options(sample_quality_item)
    basic_quality_moa.options_prompt_unlabelled = options_unlabeled
    
    # Test with no references
    messages = basic_quality_moa.get_messages_aggregator([])
    assert "You are a question-answering assistant" in messages[0]["content"]
    
    # Test with references
    messages = basic_quality_moa.get_messages_aggregator(["ref1", "ref2"])
    assert "ref1" in messages[1]["content"]
    assert "ref2" in messages[1]["content"]

def test_alpaca_aggregator_messages(basic_alpaca_moa, sample_alpaca_item):
    """Test final aggregator message creation for Alpaca"""
    basic_alpaca_moa.instruction = sample_alpaca_item["instruction"]
    
    # Test with references
    messages = basic_alpaca_moa.get_messages_aggregator(["ref1", "ref2"])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "ref1" in messages[0]["content"]
    assert "ref2" in messages[0]["content"]
    assert sample_alpaca_item["instruction"] in messages[1]["content"]

def test_deceptive_ignore_refs(basic_quality_moa, sample_quality_item):
    """Test deceptive_ignore_refs functionality"""
    # Modify moa to ignore refs for deceptive agents
    basic_quality_moa.deceptive_ignore_refs = True
    basic_quality_moa.question = sample_quality_item["question"]
    basic_quality_moa.passage = sample_quality_item["article"]

    basic_quality_moa.options_prompt_labelled_true, basic_quality_moa.options_prompt_labelled_incorrect, basic_quality_moa.options_prompt_labelled_flipped, basic_quality_moa.options_prompt_unlabelled, basic_quality_moa.incorrect_answer = basic_quality_moa.label_options(sample_quality_item)
    basic_quality_moa.subpassage_dict = basic_quality_moa.split_passage(basic_quality_moa.passage)

    
    # Get messages for deceptive model with previous references
    messages = basic_quality_moa.get_messages(1, "model2", ["prev_ref"])
    
    # Check that references are not included when deceptive_ignore_refs is True
    assert not any("prev_ref" in msg["content"] for msg in messages)

def test_full_synthesis_quality(basic_quality_moa, sample_quality_item, mocker):
    """Test full synthesis pipeline for Quality MoA"""
    # Mock generate_together function
    mocker.patch('moa_class.generate_together', return_value="mocked response")
    
    result = basic_quality_moa.full_synthesise(sample_quality_item)
    
    assert "output" in result
    assert "generator" in result
    assert "references" in result
    assert "incorrect_answer" in result
    assert result["generator"] == "agg_model" + "test_generator"

def test_full_synthesis_alpaca(basic_alpaca_moa, sample_alpaca_item, mocker):
    """Test full synthesis pipeline for Alpaca MoA"""
    # Mock generate_together function
    mocker.patch('moa_class.generate_together', return_value="mocked response")
    
    result = basic_alpaca_moa.full_synthesise(sample_alpaca_item)
    
    assert "output" in result
    assert "generator" in result
    assert "references" in result
    assert result["generator"] == "agg_model" + "test_generator"

if __name__ == "__main__":
    pytest.main([__file__])
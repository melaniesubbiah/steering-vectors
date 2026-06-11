"""
Predefined steering mode that maps dataset preferences to existing steering vectors.
"""
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PredefinedSteeringPipeline:
    """
    Pipeline that uses predefined steering vectors based on dataset preferences.
    
    Supports two modes of operation:
    1. Steering only: Uses activation steering vectors to guide model behavior
    2. Steering + Prompting: Combines steering vectors with explicit trait prompts
    
    The prompting option can be controlled via the use_prompting parameter in __init__
    or overridden per run in run_with_predefined_vectors().
    """
    
    # Mapping from dataset-specific preferences to available steering vector names
    # Each entry can be either a string (vector name) or dict with 'vector', 'layer', 'coeff' keys
    PREFERENCE_TO_VECTOR_MAP = {
        # CNN DailyMail preferences (from writing_preference_set.py)
        "adopt a step-by-step structure": {"vector": "step_by_step", "layer": 14, "coeff": 4.0},
        "include a simile": [
            {"vector": "creative", "layer": 16, "coeff": 0.8},
            {"vector": "creative", "layer": 18, "coeff": 1.0},
            {"vector": "creative", "layer": 20, "coeff": 1.2},
            {"vector": "creative", "layer": 22, "coeff": 1.0},
            {"vector": "creative", "layer": 24, "coeff": 0.8}
        ],

        "use ampersands (&) instead of \"and\"s": [
            {"vector": "creative", "layer": 18, "coeff": 0.6},
            {"vector": "creative", "layer": 20, "coeff": 0.8},
            {"vector": "creative", "layer": 22, "coeff": 1.0},
            {"vector": "creative", "layer": 24, "coeff": 0.8}
        ],
        "write in the style of a children's book": {"vector": "childlike", "layer": 10, "coeff": 2.0},
        "be sharply critical": "critical",
        "be blatantly sarcastic": "sarcastic",
        # Other dataset preferences
        "adopt a third person narrative": "narrative_coherence",
        "include rhetorical questions": "inquisitive_emoji",
        "short and punchy sentences": "short_punchy",
        "use ALLCAPS to emphasize certain words": "emotional",
        "write in the style of a tweet": [
            {"vector": "brief_immersive", "layer": 16, "coeff": 1.3},
            {"vector": "brief_immersive", "layer": 18, "coeff": 1.5},
            {"vector": "brief_immersive", "layer": 20, "coeff": 1.7},
            {"vector": "brief_immersive", "layer": 22, "coeff": 1.3},
            {"vector": "brief_immersive", "layer": 24, "coeff": 0.7}
        ],
        "adopt a rhyming structure": "rhyming_screenplay",
        "include modern slang": [
            {"vector": "humorous", "layer": 16, "coeff": 0.8},
            {"vector": "humorous", "layer": 18, "coeff": 1.0},
            {"vector": "humorous", "layer": 20, "coeff": 1.2},
            {"vector": "humorous", "layer": 22, "coeff": 1.0}
        ],
        "use semicolons (;) when possible": "analytical",
        "write in the style of a screenplay": [
            {"vector": "rhyming_screenplay", "layer": 14, "coeff": 0.6},
            {"vector": "rhyming_screenplay", "layer": 16, "coeff": 0.8},
            {"vector": "rhyming_screenplay", "layer": 18, "coeff": 1.0},
            {"vector": "rhyming_screenplay", "layer": 20, "coeff": 1.2},
            {"vector": "rhyming_screenplay", "layer": 22, "coeff": 1.0}
        ],
        "adopt a question-answering style structure": "qa_podcast",
        "include personifications": [
            {"vector": "creative", "layer": 16, "coeff": 0.7},
            {"vector": "creative", "layer": 18, "coeff": 0.9},
            {"vector": "creative", "layer": 20, "coeff": 1.1},
            {"vector": "creative", "layer": 22, "coeff": 0.9}
        ],
        "use archaic language": [
            {"vector": "creative", "layer": 15, "coeff": 0.8},
            {"vector": "creative", "layer": 17, "coeff": 1.0},
            {"vector": "creative", "layer": 19, "coeff": 1.2},
            {"vector": "creative", "layer": 21, "coeff": 1.0},
            {"vector": "creative", "layer": 23, "coeff": 0.8}
        ],
        "write in the style of a podcast": "podcast_style",
        #"adopt a second person narrative": "interactive_playful",
        "include onomatopoeias": [
            {"vector": "onomatopoeia", "layer": 8, "coeff": 0.75},
        ],
        "use imagery": [
            {"vector": "vivid_imagery", "layer": 10, "coeff": 2.0},
        ],
        "write in the style of old timey radio": "old_timey_radio",
        
        # General preferences that might appear
        "professional": "analytical",
        "casual": "humorous", 
        "interactive": "interactive_playful",
        "playful": "interactive_playful",
        "positive": "optimistic",
        "brief": "brief_immersive",
        "bullet points": "bullet_parallel",
        "parallel structure": "bullet_parallel",
        "structured": "header_structured"
    }
    
    # Known dataset preference mappings (from ARTICLE_SOURCE_TO_PREFERENCE)
    DATASET_PREFERENCES = {
        "cnn_dailymail": {
            "plume": ["write in the style of a children's book"],
            "prelude": ["interactive", "playful", "positive", "brief", "structured"]
        },
        "slf5k": {
            "plume": ["write in the style of a tweet"],
            "prelude": ["brief", "interactive", "positive"]
        },
        "wikipedia": {
            "plume": ["write the style of a screenplay"],
            "prelude": ["brief", "bullet points", "parallel structure"]
        },
        "imdb": {
            "plume": ["write in the style of old timey radio"],
            "prelude": ["interactive"]
        },
        "CShorten/ML-ArXiv-Papers": {
            "plume": ["write in the style of a podcast"],
            "prelude": ["brief", "bullet points", "parallel structure"]
        },
        "ccby": {
            "plume": ["be highly inquisitive"],
            "prelude": ["interactive", "playful", "positive", "brief", "structured"]
        },
        "paper_tweet": {
            "plume": ["be blatantly sarcastic"],
            "prelude": ["brief", "interactive", "positive"]
        },
        "ampere": {
            "plume": ["short and punchy sentences"],
            "prelude": ["brief", "interactive", "positive"]
        }
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        vector_dir: str = "pretrained_vectors",
        framework: str = "plume",
        use_prompting: bool = False,
        enable_evaluation: bool = True,
        csv_mode: bool = False
    ):
        from persona_vectors.model_loader import ModelLoader
        self.model_loader = ModelLoader(
            cache_dir=None,
            device=None if device == "auto" else device
        )

        # For compatibility, create a minimal service object that has the generate methods
        from use_pretrained_vectors import PretrainedVectorPipeline
        self.service = PretrainedVectorPipeline(
            vector_dir=str(vector_dir),
            model_name=model_name,
            inference_method="hidden_state_similarity"
        )
        self.framework = framework
        self.vector_dir = Path(vector_dir)
        self.use_prompting = use_prompting
        self.enable_evaluation = enable_evaluation
        self.csv_mode = csv_mode
        
        # Determine model subfolder
        if "qwen" in model_name.lower():
            self.vector_dir = self.vector_dir / "qwen"
        elif "llama" in model_name.lower():
            self.vector_dir = self.vector_dir / "llama"
        else:
            self.vector_dir = self.vector_dir / "qwen"
        
        self.available_vectors = self._scan_available_vectors()
        self._setup_evaluator()
        print(f"Predefined steering mode: Found {len(self.available_vectors)} vectors")
    
    def _scan_available_vectors(self) -> Dict[str, Path]:
        """Scan directory for .pt vector files."""
        vectors = {}
        if not self.vector_dir.exists():
            print(f"Vector directory {self.vector_dir} not found")
            return vectors
        
        for pt_file in self.vector_dir.glob("*.pt"):
            trait_name = pt_file.stem
            # Remove common suffixes
            for suffix in ['_response_avg_diff', '_prompt_avg_diff', '_vector']:
                if trait_name.endswith(suffix):
                    trait_name = trait_name.replace(suffix, '')
                    break
            vectors[trait_name] = pt_file
        
        return vectors
    
    def _setup_evaluator(self):
        """Initialize ml-predict evaluator."""
        print(f"[DEBUG] Setting up evaluator: enable_evaluation={self.enable_evaluation}")
        if not self.enable_evaluation:
            print("⚠ Evaluation disabled via enable_evaluation=False")
            self.evaluator = None
            return

        try:
            # Add ml-predict to path
            from pathlib import Path
            import sys
            mlp_root = Path(__file__).resolve().parents[1] / 'persona_vectors' / 'ml-predict'
            if str(mlp_root) not in sys.path:
                sys.path.append(str(mlp_root))
            
            from preference_inferrer.tasks.writing.writing_quality_assesor import WritingQualityAssessor
            from preference_inferrer.common.configurations import configure_config
            from types import SimpleNamespace
            import logging
            
            # Create config dict and convert to properly configured namespace
            config_dict = {
                "bertscore_model": "microsoft/deberta-xlarge-mnli",
                "logging_level": "info",
                "base_dir": None,  # Will use current directory
                "user": {
                    "llm_name": "gpt-4o-mini"
                }
            }
            config = configure_config(config_dict)
            # Ensure required fields for preference matching exist
            if not hasattr(config, 'gpt_temperature'):
                config.gpt_temperature = 0.01
            if not hasattr(config, 'seed'):
                config.seed = 1352
            
            self.evaluator = WritingQualityAssessor(config)
            print("✓ Evaluation metrics enabled")
        except Exception as e:
            print(f"⚠ Could not load evaluator, skipping metrics: {e}")
            self.evaluator = None
    
    def _compute_evaluation_metrics(self,
                                   generated_text: str,
                                   reference_text: str,
                                   preferences: List[str],
                                   task_name: str,
                                   inferred_preferences: List[str] = None) -> Dict[str, float]:
        """Compute evaluation metrics using ml-predict evaluator."""
        # Initialize all metrics to ensure consistent columns (use float64 NaN)
        import numpy as np
        metrics = {
            'bertscore_f1': np.float64('nan'),
            'bertscore_precision': np.float64('nan'),
            'bertscore_recall': np.float64('nan'),
            'levenshtein_distance': np.float64('nan'),
            'normalized_levenshtein': np.float64('nan'),
            'per_preference_match': np.float64('nan'),
            'full_preference_match': np.float64('nan'),
            'average_preference_match': np.float64('nan'),
            'preference_similarity': np.float64('nan')
        }
        
        print(f"[DEBUG] Computing evaluation metrics: evaluator={self.evaluator is not None}, text_len={len(generated_text) if generated_text else 0}")
        if not self.evaluator or not generated_text:
            print(f"[DEBUG] Skipping evaluation: evaluator={self.evaluator is not None}, has_text={bool(generated_text)}")
            return metrics
        
        try:
            # Reference-dependent metrics (only if reference exists)
            if reference_text and reference_text.strip():
                print(f"Computing reference metrics: ref_len={len(reference_text)}, gen_len={len(generated_text)}")
                
                # BERTScore
                try:
                    f1, precision, recall = self.evaluator.get_bertscore(
                        predictions=generated_text,
                        references=reference_text
                    )
                    print(f"BERTScore result: f1={f1}, precision={precision}, recall={recall}")
                    metrics['bertscore_f1'] = np.float64(f1)
                    metrics['bertscore_precision'] = np.float64(precision)
                    metrics['bertscore_recall'] = np.float64(recall)
                except Exception as e:
                    print(f"BERTScore error: {e}")
                
                # Levenshtein distance
                try:
                    ldist, n_ldist = self.evaluator.calculate_levenshtein_distance(
                        agent_completion=generated_text,
                        user_completion=reference_text
                    )
                    print(f"Levenshtein: {ldist}, normalized: {n_ldist}")
                    metrics['levenshtein_distance'] = np.float64(ldist)
                    metrics['normalized_levenshtein'] = np.float64(n_ldist)
                except Exception as e:
                    print(f"Levenshtein error: {e}")
            
            # Preference metrics (only if preferences available)
            if preferences and any(p.strip() for p in preferences):
                try:
                    from preference_inferrer.tasks.task_instances import WritingTaskInstance
                    from preference_inferrer.preference_sets.writing_preference_set import WritingPreferenceSet
                    from types import SimpleNamespace
                    
                    pref_list = [p.strip() for p in preferences if p.strip()]
                    
                    # Create a WritingTaskInstance
                    task_instance = WritingTaskInstance(
                        task_type=task_name,
                        task_content="",  # We don't have the original task content
                        source="unknown",  # We don't have the source
                        source_idx=0
                    )
                    
                    # Set the completions manually
                    task_instance.agent_completion = generated_text
                    task_instance.user_completion = reference_text or ""
                    
                    # Create config for WritingPreferenceSet (not strictly needed by evaluator)
                    if hasattr(self.evaluator, 'config'):
                        # Ensure evaluator config has required attributes
                        if not hasattr(self.evaluator.config, 'gpt_temperature'):
                            self.evaluator.config.gpt_temperature = 0.01
                        if not hasattr(self.evaluator.config, 'seed'):
                            self.evaluator.config.seed = 1352
                        import copy
                        pref_config = copy.deepcopy(self.evaluator.config)
                    else:
                        pref_config = SimpleNamespace()
                        pref_config.gpt_temperature = 0.01
                        pref_config.seed = 1352
                    
                    # Create WritingPreferenceSet
                    preference_set = WritingPreferenceSet(pref_list, pref_config)
                    
                    # Per-preference component match
                    ppm_score = self.evaluator.llm_judges_preference_matching(
                        task_instance=task_instance,
                        true_preference_set=preference_set,
                        per_preference=True
                    )
                    metrics['per_preference_match'] = np.float64(ppm_score)
                    
                    # Full preference match (compound)
                    fpm_score = self.evaluator.llm_judges_preference_matching(
                        task_instance=task_instance,
                        true_preference_set=preference_set,
                        per_preference=False
                    )
                    metrics['full_preference_match'] = np.float64(fpm_score)
                    
                    # Average match
                    metrics['average_preference_match'] = np.float64((ppm_score + fpm_score) / 2.0)
                    
                    print(f"Preference scores: ppm={ppm_score}, fpm={fpm_score}, avg={metrics['average_preference_match']}")

                    # Preference similarity (compare inferred vs true preferences)
                    if inferred_preferences and inferred_preferences != preferences:
                        try:
                            true_prefs_str = "; ".join(preferences)
                            inferred_prefs_str = "; ".join(inferred_preferences)

                            similarity_score = self.evaluator.score_preference_similarity(
                                pred_preference=inferred_prefs_str,
                                true_preference=true_prefs_str
                            )
                            metrics['preference_similarity'] = np.float64(similarity_score)
                            print(f"Preference similarity: {similarity_score} (inferred: {inferred_prefs_str[:50]}... vs true: {true_prefs_str[:50]}...)")
                        except Exception as e:
                            print(f"Preference similarity error: {e}")

                except Exception as e:
                    print(f"Preference matching error: {e}")
            
            # Ensure all values are float64 (convert None to NaN if any snuck in)
            import numpy as np
            for key, value in metrics.items():
                if value is None:
                    metrics[key] = np.float64('nan')
                elif not isinstance(value, (int, float)):
                    metrics[key] = np.float64('nan')
                else:
                    metrics[key] = np.float64(value)
            
            return metrics
            
        except Exception as e:
            print(f"⚠ Error computing metrics: {e}")
            # Return float64 NaN values to maintain consistent typing
            import numpy as np
            return {key: np.float64('nan') for key in metrics.keys()}

    def get_dataset_preferences(self, dataset_name: str) -> List[str]:
        """
        Get predefined preferences for a specific dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'cnn_dailymail')

        Returns:
            List of preference strings for the dataset
        """
        if dataset_name not in self.DATASET_PREFERENCES:
            return []

        return self.DATASET_PREFERENCES[dataset_name].get(self.framework, [])

    def _generate_layer_vector_similarity_analysis(self, dataset: str, texts: List[str], inferrer, output_dir: Path) -> pd.DataFrame:
        """
        Generate comprehensive analysis of cosine similarity between demo texts
        and all available vectors at all layers.
        """

        print(f"Generating layer-vector similarity analysis for {dataset}...")

        analysis_rows = []

        # Process each demo text
        for text_idx, text in enumerate(texts):
            print(f"  Processing demo {text_idx + 1}/{len(texts)} (length: {len(text)} chars)")

            try:
                # Extract hidden states for this text
                hidden_states = inferrer._extract_hidden_states(text)

                # Process each available vector
                for vector_name, vector_path in self.available_vectors.items():
                    try:
                        # Load the vector
                        vector = torch.load(vector_path, map_location='cpu')

                        if vector.ndim == 2:
                            # 2D vector: [num_layers, hidden_dim]
                            num_layers = vector.shape[0]

                            # Test similarity at each layer the vector supports
                            for layer_idx in range(num_layers):
                                if layer_idx in hidden_states:
                                    layer_vector = vector[layer_idx]
                                    hidden_state = hidden_states[layer_idx]

                                    # Calculate cosine similarity
                                    similarity = inferrer._cosine_similarity(hidden_state, layer_vector)

                                    analysis_rows.append({
                                        'dataset': dataset,
                                        'demo_idx': text_idx,
                                        'demo_text_preview': text[:100] + '...' if len(text) > 100 else text,
                                        'demo_length': len(text),
                                        'vector_name': vector_name,
                                        'layer': layer_idx,
                                        'cosine_similarity': similarity,
                                        'vector_shape': f"{vector.shape}",
                                        'hidden_state_shape': f"{hidden_state.shape}"
                                    })

                        elif vector.ndim == 1:
                            # 1D vector: test against all available hidden state layers
                            for layer_idx, hidden_state in hidden_states.items():
                                similarity = inferrer._cosine_similarity(hidden_state, vector)

                                analysis_rows.append({
                                    'dataset': dataset,
                                    'demo_idx': text_idx,
                                    'demo_text_preview': text[:100] + '...' if len(text) > 100 else text,
                                    'demo_length': len(text),
                                    'vector_name': vector_name,
                                    'layer': layer_idx,
                                    'cosine_similarity': similarity,
                                    'vector_shape': f"{vector.shape}",
                                    'hidden_state_shape': f"{hidden_state.shape}"
                                })

                    except Exception as e:
                        print(f"    Error processing vector {vector_name}: {e}")

            except Exception as e:
                print(f"  Error processing demo {text_idx}: {e}")

        # Create DataFrame
        analysis_df = pd.DataFrame(analysis_rows)

        if not analysis_df.empty:
            # Save the analysis
            safe_name = dataset.replace('/', '_').replace('-', '_')
            analysis_path = output_dir / f"{safe_name}_layer_vector_similarity_analysis.csv"
            analysis_df.to_csv(analysis_path, index=False)
            print(f"Saved layer-vector similarity analysis: {analysis_path}")
            print(f"Analysis shape: {analysis_df.shape}")

            # Print summary stats
            print("Top 10 highest similarities:")
            top_similarities = analysis_df.nlargest(10, 'cosine_similarity')
            for _, row in top_similarities.iterrows():
                print(f"  {row['vector_name']} @ layer {row['layer']}: {row['cosine_similarity']:.4f}")

        return analysis_df

    def generate_comprehensive_demo_similarity_analysis(self, inferrer, output_dir: Path) -> pd.DataFrame:
        """
        Generate comprehensive analysis across ALL dataset demos and all vectors/layers.
        """
        print("Generating comprehensive demo similarity analysis across all datasets...")

        all_analysis_rows = []

        # Get all available dataset demos
        available_datasets = list(inferrer.dataset_demos.keys())
        print(f"Found {len(available_datasets)} dataset demos: {available_datasets}")

        for dataset in available_datasets:
            demo_text = inferrer.dataset_demos[dataset]
            print(f"\nProcessing dataset demo: {dataset} (length: {len(demo_text)} chars)")

            try:
                # Extract hidden states for this demo
                hidden_states = inferrer._extract_hidden_states(demo_text)

                # Process each available vector
                for vector_name, vector_path in self.available_vectors.items():
                    try:
                        # Load the vector
                        vector = torch.load(vector_path, map_location='cpu')

                        if vector.ndim == 2:
                            # 2D vector: [num_layers, hidden_dim]
                            num_layers = vector.shape[0]

                            # Test similarity at each layer the vector supports
                            for layer_idx in range(num_layers):
                                if layer_idx in hidden_states:
                                    layer_vector = vector[layer_idx]
                                    hidden_state = hidden_states[layer_idx]

                                    # Calculate cosine similarity
                                    similarity = inferrer._cosine_similarity(hidden_state, layer_vector)

                                    all_analysis_rows.append({
                                        'dataset': dataset,
                                        'demo_text_preview': demo_text[:200] + '...' if len(demo_text) > 200 else demo_text,
                                        'demo_length': len(demo_text),
                                        'vector_name': vector_name,
                                        'layer': layer_idx,
                                        'cosine_similarity': similarity,
                                        'vector_shape': f"{vector.shape}",
                                        'vector_type': '2D_multilayer'
                                    })

                        elif vector.ndim == 1:
                            # 1D vector: test against all available hidden state layers
                            for layer_idx, hidden_state in hidden_states.items():
                                similarity = inferrer._cosine_similarity(hidden_state, vector)

                                all_analysis_rows.append({
                                    'dataset': dataset,
                                    'demo_text_preview': demo_text[:200] + '...' if len(demo_text) > 200 else demo_text,
                                    'demo_length': len(demo_text),
                                    'vector_name': vector_name,
                                    'layer': layer_idx,
                                    'cosine_similarity': similarity,
                                    'vector_shape': f"{vector.shape}",
                                    'vector_type': '1D_single_layer'
                                })

                    except Exception as e:
                        print(f"    Error processing vector {vector_name}: {e}")

            except Exception as e:
                print(f"  Error processing dataset {dataset}: {e}")

        # Create comprehensive DataFrame
        comprehensive_df = pd.DataFrame(all_analysis_rows)

        if not comprehensive_df.empty:
            # Save the comprehensive analysis
            analysis_path = output_dir / "comprehensive_demo_vector_similarity_analysis.csv"
            comprehensive_df.to_csv(analysis_path, index=False)
            print(f"\nSaved comprehensive demo similarity analysis: {analysis_path}")
            print(f"Analysis shape: {comprehensive_df.shape}")

            # Print summary stats
            print("\nDataset coverage:")
            dataset_counts = comprehensive_df['dataset'].value_counts()
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count} vector-layer combinations")

            print("\nTop 15 highest similarities across all datasets:")
            top_similarities = comprehensive_df.nlargest(15, 'cosine_similarity')
            for _, row in top_similarities.iterrows():
                print(f"  {row['dataset']} - {row['vector_name']} @ layer {row['layer']}: {row['cosine_similarity']:.4f}")

            # Group by vector and show best performing dataset for each
            print("\nBest dataset match per vector (top layer):")
            vector_best = comprehensive_df.loc[comprehensive_df.groupby('vector_name')['cosine_similarity'].idxmax()]
            for _, row in vector_best.iterrows():
                print(f"  {row['vector_name']}: {row['dataset']} @ layer {row['layer']} ({row['cosine_similarity']:.4f})")

        return comprehensive_df

    def infer_persona_from_texts(self, texts, dataset=None):
        """Wrapper method to delegate to the service."""
        return self.service.infer_persona_from_texts(texts, dataset=dataset)

    def generate_batch_with_optional_steering(self, **kwargs):
        """Wrapper method to delegate to the service."""
        return self.service.generate_batch_with_optional_steering(**kwargs)

    def get_steering_vectors_for_dataset(self, dataset_name: str) -> Tuple[List[str], List[Dict]]:
        """
        Get predefined steering vectors for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cnn_dailymail')
            
        Returns:
            Tuple of (preference_list, steering_instructions_list)
            Each steering instruction is a dict with keys: steering_vector, layer_idx, coeff, positions
        """
        # Get preferences for this dataset and framework
        if dataset_name not in self.DATASET_PREFERENCES:
            print(f"No predefined preferences for dataset: {dataset_name}")
            return [], {}
        
        preferences = self.DATASET_PREFERENCES[dataset_name].get(self.framework, [])
        if not preferences:
            print(f"No preferences for {dataset_name} with framework {self.framework}")
            return [], {}
        
        print(f"Using predefined preferences for {dataset_name}: {preferences}")
        
        # Map preferences to steering instructions
        steering_instructions = []
        mapped_preferences = []
        
        for pref in preferences:
            if pref in self.PREFERENCE_TO_VECTOR_MAP:
                vector_config = self.PREFERENCE_TO_VECTOR_MAP[pref]
                
                # Handle string, dict, and list formats
                configs_to_process = []
                if isinstance(vector_config, str):
                    configs_to_process = [{
                        "vector": vector_config,
                        "layer": 20,  # default layer
                        "coeff": 2.0  # default coeff
                    }]
                elif isinstance(vector_config, dict):
                    configs_to_process = [vector_config]
                elif isinstance(vector_config, list):
                    configs_to_process = vector_config
                else:
                    print(f"  ✗ Invalid vector config for preference: {pref}")
                    continue
                
                # Process each config (for multi-layer steering)
                for config in configs_to_process:
                    vector_name = config["vector"]
                    layer_idx = config.get("layer", 20)
                    coeff = config.get("coeff", 2.0)
                    
                    if vector_name in self.available_vectors:
                        try:
                            vector_path = self.available_vectors[vector_name]
                            vector = torch.load(vector_path, map_location='cpu')
                            
                            # Handle 2D vectors [num_layers, hidden_size] by extracting the layer
                            if vector.ndim == 2:
                                if layer_idx < vector.shape[0]:
                                    vector = vector[layer_idx]  # Extract specific layer
                                else:
                                    vector = vector[-1]  # Use last layer if layer_idx too high
                            elif vector.ndim != 1:
                                print(f"  ✗ Unsupported vector shape {vector.shape} for {vector_name}")
                                continue
                            
                            instruction = {
                                "steering_vector": vector,
                                "layer_idx": layer_idx,
                                "coeff": coeff,
                                "positions": "response"  # default to response steering
                            }
                            
                            steering_instructions.append(instruction)
                            print(f"  ✓ Mapped '{pref}' -> {vector_name} (layer={layer_idx}, coeff={coeff}, shape={vector.shape})")
                        except Exception as e:
                            print(f"  ✗ Failed to load {vector_name}: {e}")
                    else:
                        print(f"  ✗ Vector not found: {vector_name} (for preference: {pref})")
                
                # Add preference to mapped list only once per preference
                if any(config["vector"] in self.available_vectors for config in configs_to_process):
                    mapped_preferences.append(pref)
            else:
                print(f"  ✗ No vector mapping for preference: {pref}")
        
        return mapped_preferences, steering_instructions
    
    def _create_trait_prompting_text(self, preferences: List[str]) -> str:
        """Create prompting text that encourages the desired traits."""
        if not preferences:
            return ""
        
        trait_descriptions = {
            "adopt a step-by-step structure": "structure your response in clear, numbered steps",
            "include a simile": "include at least one simile comparison",
            "use ampersands (&) instead of \"and\"s": "use ampersands (&) instead of the word 'and'",
            "write in the style of a children's book": "write in a simple, engaging style suitable for children",
            "adopt a third person narrative": "write using third person perspective (he/she/they)",
            "include rhetorical questions": "include thought-provoking rhetorical questions",
            "use ALLCAPS to emphasize certain words": "use ALL CAPITAL LETTERS to emphasize important words",
            "write in the style of a tweet": "write concisely as if posting on social media",
            "adopt a rhyming structure": "include rhyming elements in your response",
            "include modern slang": "incorporate contemporary slang and informal language",
            "use semicolons (;) when possible": "use semicolons to connect related clauses",
            "write in the style of a screenplay": "format your response like a screenplay with dialogue and stage directions",
            "adopt a question-answering style structure": "structure your response as questions followed by answers",
            "include personifications": "give human characteristics to non-human elements",
            "use archaic language": "use old-fashioned or archaic language and expressions",
            "write in the style of a podcast": "write as if speaking on a podcast with conversational tone",
            "adopt a second person narrative": "write using second person perspective (you)",
            "include onomatopoeias": "include sound words like 'boom', 'crash', 'whisper'",
            "use imagery": "include vivid descriptive language that appeals to the senses",
            "write in the style of old timey radio": "use dramatic, theatrical language and delivery style reminiscent of 1940s radio broadcasts, focusing on tone and presentation rather than content",
            "professional": "maintain a professional and formal tone",
            "casual": "use a relaxed and informal conversational style",
            "interactive": "engage the reader with questions and interactive elements",
            "playful": "adopt a fun and lighthearted tone",
            "positive": "maintain an optimistic and upbeat perspective",
            "brief": "keep your response concise and to the point",
            "bullet points": "organize information using bullet points",
            "parallel structure": "use consistent grammatical structure across similar elements",
            "structured": "organize your response with clear headings and sections"
        }
        
        matched_descriptions = []
        for pref in preferences:
            if pref in trait_descriptions:
                matched_descriptions.append(trait_descriptions[pref])
        
        if not matched_descriptions:
            return ""
        
        prompt_text = "Please " + ", ".join(matched_descriptions) + "."
        return prompt_text
    
    def run_comparative_evaluation(
        self,
        datasets: List[str],
        num_examples: int = 25,
        output_dir: str = "results_comparative",
        hf_repo: Optional[str] = None,
        steering_layer: int = 20,
        steering_coeff: float = 2.0,
        steering_type: str = "response"
    ) -> Dict[str, any]:
        """
        Run comprehensive comparative evaluation with LLM judge.
        
        This generates results for all four conditions:
        1. Baseline: No prompting, no steering
        2. Prompting-only: Trait descriptions in prompt, no steering
        3. Steering-only: No trait descriptions, only activation steering
        4. Prompting+steering: Both trait descriptions and activation steering
        
        All approaches are evaluated with LLM judge for preference matching.
        """
        from data.mlp_loader import MLPDatasetLoader
        from prose.task_prompts import get_task_prompt_builder
        from utils.hf_upload import push_parquet_results_to_hf
        import pandas as pd
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        loader = MLPDatasetLoader()
        
        for dataset in datasets:
            print(f"\n=== Comparative Evaluation for {dataset} ===")
            
            # Get predefined steering instructions for this dataset
            preferences, steering_instructions = self.get_steering_vectors_for_dataset(dataset)
            
            if not steering_instructions:
                print(f"Skipping {dataset} - no steering instructions available")
                continue
            
            # Load data
            items = loader.load_examples_with_metadata([dataset], num_examples)
            if not items:
                print(f"Skipping {dataset} - no data loaded")
                continue
            
            texts = [item['text'] for item in items]
            references = [item.get('reference', '') for item in items]
            task_name = items[0].get('task_name', 'summarization') if items else 'summarization'
            
            # Get actual trait names from steering instructions instead of layer numbers
            trait_names = []
            for inst in steering_instructions:
                # Find the vector name that corresponds to this steering instruction
                for pref in preferences:
                    vector_config = self.PREFERENCE_TO_VECTOR_MAP.get(pref)
                    if isinstance(vector_config, list):
                        for config in vector_config:
                            if config.get("layer") == inst["layer_idx"]:
                                trait_names.append(config["vector"])
                                break
                    elif isinstance(vector_config, dict):
                        if vector_config.get("layer") == inst["layer_idx"]:
                            trait_names.append(vector_config["vector"])
                    elif isinstance(vector_config, str):
                        trait_names.append(vector_config)
            
            # Remove duplicates while preserving order
            unique_trait_names = []
            for name in trait_names:
                if name not in unique_trait_names:
                    unique_trait_names.append(name)
            
            trait_prompt = self._create_trait_prompting_text(preferences)
            
            print(f"Using preferences: {preferences}")
            print(f"Using trait prompt: {trait_prompt}")
            print(f"Using trait names: {unique_trait_names}")
            
            # Get base prompt builder
            base_prompt_builder = get_task_prompt_builder(dataset, task_name)
            
            # Create prompting-enhanced prompt builder
            def prompting_prompt_builder(text):
                base_prompt = base_prompt_builder(text)
                if trait_prompt:
                    return f"{base_prompt}\n\n{trait_prompt}"
                return base_prompt
            
            print(f"\n1. Generating baseline outputs (no prompting, no steering)...")
            baseline_results = self.service.generate_batch_with_optional_steering(
                texts=texts,
                task_prompt_builder=base_prompt_builder,  # No trait prompting
                steering_vectors={},  # No steering vectors
                layer=steering_layer,
                coeff=steering_coeff,
                steering_type=steering_type
            )
            
            print(f"2. Generating prompting-only outputs...")
            prompting_only_results = self.service.generate_batch_with_optional_steering(
                texts=texts,
                task_prompt_builder=prompting_prompt_builder,  # With trait prompting
                steering_vectors={},  # No steering vectors
                layer=steering_layer,
                coeff=steering_coeff,
                steering_type=steering_type
            )
            
            print(f"3. Generating steering-only outputs...")
            steering_only_results = self.service.generate_batch_with_multiple_steering(
                texts=texts,
                task_prompt_builder=base_prompt_builder,  # No trait prompting
                steering_instructions=steering_instructions  # With steering
            )
            
            print(f"4. Generating prompting+steering outputs...")
            prompting_steering_results = self.service.generate_batch_with_multiple_steering(
                texts=texts,
                task_prompt_builder=prompting_prompt_builder,  # With trait prompting
                steering_instructions=steering_instructions  # With steering
            )
            
            # Create comparative results dataframe
            rows = []
            for i in range(len(texts)):
                # Get results from all approaches
                baseline_result = baseline_results[i] if i < len(baseline_results) else {}
                prompting_result = prompting_only_results[i] if i < len(prompting_only_results) else {}
                steering_result = steering_only_results[i] if i < len(steering_only_results) else {}
                combined_result = prompting_steering_results[i] if i < len(prompting_steering_results) else {}
                
                # Extract outputs
                baseline_output = baseline_result.get("steered_output") or baseline_result.get("baseline_output", "")
                prompting_only_output = prompting_result.get("steered_output") or prompting_result.get("baseline_output", "")
                steering_only_output = steering_result.get("steered_output", "")
                prompting_steering_output = combined_result.get("steered_output", "")
                
                human_answer = references[i] if i < len(references) else ""
                
                # Evaluate all approaches
                baseline_metrics = self._compute_evaluation_metrics(
                    generated_text=baseline_output,
                    reference_text=human_answer,
                    preferences=preferences,
                    task_name=task_name
                )
                
                prompting_metrics = self._compute_evaluation_metrics(
                    generated_text=prompting_only_output,
                    reference_text=human_answer,
                    preferences=preferences,
                    task_name=task_name
                )
                
                steering_metrics = self._compute_evaluation_metrics(
                    generated_text=steering_only_output,
                    reference_text=human_answer,
                    preferences=preferences,
                    task_name=task_name
                )
                
                combined_metrics = self._compute_evaluation_metrics(
                    generated_text=prompting_steering_output,
                    reference_text=human_answer,
                    preferences=preferences,
                    task_name=task_name
                )
                
                # Create row with all approaches
                row = {
                    "dataset": dataset,
                    "example_idx": i,
                    "query": (baseline_result.get("prompt", "") or prompting_result.get("prompt", "") or 
                             steering_result.get("prompt", "") or combined_result.get("prompt", "")),
                    "example": texts[i],
                    "human_answer": human_answer,
                    "preferences": "; ".join(preferences) if preferences else "",
                    "trait_prompt": trait_prompt,
                    "steering_vectors": "; ".join(unique_trait_names) if unique_trait_names else "",
                    
                    # All four approaches
                    "baseline_output": baseline_output,
                    "prompting_only_output": prompting_only_output,
                    "steering_only_output": steering_only_output,
                    "prompting_steering_output": prompting_steering_output,
                }
                
                # Add metrics for all approaches with prefixes
                for metric, value in baseline_metrics.items():
                    row[f"baseline_{metric}"] = value
                
                for metric, value in prompting_metrics.items():
                    row[f"prompting_only_{metric}"] = value
                
                for metric, value in steering_metrics.items():
                    row[f"steering_only_{metric}"] = value
                
                for metric, value in combined_metrics.items():
                    row[f"prompting_steering_{metric}"] = value
                
                rows.append(row)
            
            # Save comparative results
            df = pd.DataFrame(rows)
            safe_name = dataset.replace('/', '_')
            parquet_path = output_dir / f"{safe_name}_comparative_results.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"Saved {len(rows)} comparative results to {parquet_path}")
            
            results[dataset] = {
                'preferences': preferences,
                'steering_instructions': unique_trait_names,
                'num_examples': len(texts),
                'output_file': str(parquet_path),
                'trait_prompt': trait_prompt
            }
        
        # Push to HuggingFace if requested
        if hf_repo and results:
            try:
                url = push_parquet_results_to_hf(hf_repo, str(output_dir), repo_type='dataset')
                print(f"\nPushed comparative results to HuggingFace: {url}")
            except Exception as e:
                print(f"Failed to push to HuggingFace: {e}")
        
        return results

    def run_with_oracle_preference_adaptive_steering(
        self,
        datasets: List[str],
        inferrer,  # HiddenStateSimilarityInferrer
        num_examples: int = 25,
        output_dir: str = "results_oracle_adaptive",
        hf_repo: Optional[str] = None,
        steering_layer: int = 20,
        steering_coeff: float = 2.0,
        steering_type: str = "response",
        mythos_data_dir: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Run pipeline using oracle preferences (predefined) combined with adaptive
        coefficient optimization from hidden state similarity.

        This approach:
        1. Uses predefined dataset preferences as "oracle" knowledge
        2. For each predefined trait, uses hidden state similarity to find optimal layer/coeff
        3. Combines the reliability of oracle preferences with adaptive optimization
        """
        from data.mlp_loader import MLPDatasetLoader
        from prose.task_prompts import get_task_prompt_builder
        from utils.hf_upload import push_parquet_results_to_hf
        from pathlib import Path
        import pandas as pd
        import torch

        print(f"=== Oracle Preference + Adaptive Steering Pipeline ===")
        print(f"Datasets: {datasets}")
        print(f"Using predefined preferences with adaptive layer/coeff optimization")
        print(f"Base steering: layer={steering_layer}, coeff={steering_coeff}, type={steering_type}")

        loader = MLPDatasetLoader(mythos_data_dir=mythos_data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        results = {}

        # Generate comprehensive similarity analysis across ALL dataset demos
        print("=" * 60)
        comprehensive_analysis = self.generate_comprehensive_demo_similarity_analysis(inferrer, output_dir)
        print("=" * 60)

        for dataset in datasets:
            print(f"\n{'='*50}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*50}")

            # Load data
            items = loader.load_examples_with_metadata([dataset], num_examples)
            if not items:
                print(f"No items loaded for {dataset}, skipping...")
                continue

            # Get oracle preferences for this dataset
            preferences = self.get_dataset_preferences(dataset)
            if not preferences:
                print(f"No predefined preferences for {dataset}, skipping...")
                continue

            print(f"Oracle preferences: {preferences}")

            texts = [item['text'] for item in items]
            references = [item.get('reference', '') for item in items]
            task_name = items[0].get('task_name', 'summarization') if items else 'summarization'

            # Generate comprehensive layer-vector-demo similarity analysis using dataset demos
            demo_text = inferrer.dataset_demos.get(dataset, "")
            if demo_text:
                similarity_analysis = self._generate_layer_vector_similarity_analysis(
                    dataset, [demo_text], inferrer, output_dir
                )
            else:
                print(f"No demo text found for {dataset}, skipping similarity analysis")

            # For each predefined preference, use hidden state similarity to find optimal steering
            optimized_steering_instructions = []

            for pref in preferences:
                print(f"\nOptimizing steering for preference: '{pref}'")

                # Get the predefined vector mapping
                vector_config = self.PREFERENCE_TO_VECTOR_MAP.get(pref)
                if not vector_config:
                    print(f"  No vector mapping found for '{pref}', skipping...")
                    continue

                # Extract trait names from the config
                if isinstance(vector_config, str):
                    trait_names = [vector_config]
                elif isinstance(vector_config, dict):
                    trait_names = [vector_config["vector"]]
                elif isinstance(vector_config, list):
                    trait_names = [config["vector"] for config in vector_config]
                else:
                    print(f"  Unknown vector config format for '{pref}', skipping...")
                    continue

                # Use hidden state similarity to get raw similarity scores for these traits
                try:
                    traits_with_scores = inferrer.infer_traits_with_scores_from_dataset(dataset)
                    trait_scores = {trait: score for trait, score in traits_with_scores}

                    # Calculate similarities for hidden state analysis
                    demo_text = inferrer.dataset_demos.get(dataset, "")
                    if demo_text:
                        hidden_states = inferrer._extract_hidden_states(demo_text)
                        raw_similarities = inferrer._compute_similarity_with_personas(hidden_states)
                    else:
                        raw_similarities = {}

                except Exception as e:
                    print(f"  Error computing similarities: {e}")
                    trait_scores = {}
                    raw_similarities = {}

                # For each trait in this preference, find optimal layer and coefficient
                for trait_name in trait_names:
                    if trait_name not in self.available_vectors:
                        print(f"    Vector not found: {trait_name}")
                        continue

                    # Load the vector to determine available layers
                    vector_path = self.available_vectors[trait_name]
                    vector = torch.load(vector_path, map_location='cpu')

                    # Get raw similarity for adaptive coefficient calculation
                    raw_similarity = raw_similarities.get(trait_name, 0.0)

                    if isinstance(vector_config, list):
                        # Use the predefined layer configurations but with adaptive coefficients
                        for config in vector_config:
                            if config["vector"] == trait_name:
                                adaptive_coeff = inferrer.calculate_adaptive_coefficient(
                                    raw_similarity,
                                    base_coeff=config["coeff"],
                                    trait_name=trait_name
                                )

                                # Handle 2D vectors by extracting the specific layer
                                layer_vector = vector
                                if vector.ndim == 2:
                                    layer_idx = config["layer"]
                                    if layer_idx < vector.shape[0]:
                                        layer_vector = vector[layer_idx]
                                    else:
                                        layer_vector = vector[-1]

                                optimized_steering_instructions.append({
                                    "steering_vector": layer_vector,
                                    "layer_idx": config["layer"],
                                    "coeff": adaptive_coeff,
                                    "positions": steering_type
                                })
                                print(f"    ✓ {trait_name} @ layer {config['layer']}: coeff {config['coeff']:.2f} → {adaptive_coeff:.3f}")
                    else:
                        # Single vector config - use adaptive coefficient with best layer
                        base_coeff = vector_config.get("coeff", steering_coeff) if isinstance(vector_config, dict) else steering_coeff
                        best_layer = vector_config.get("layer", steering_layer) if isinstance(vector_config, dict) else steering_layer

                        adaptive_coeff = inferrer.calculate_adaptive_coefficient(
                            raw_similarity,
                            base_coeff=base_coeff,
                            trait_name=trait_name
                        )

                        # Handle 2D vectors
                        layer_vector = vector
                        if vector.ndim == 2:
                            if best_layer < vector.shape[0]:
                                layer_vector = vector[best_layer]
                            else:
                                layer_vector = vector[-1]

                        optimized_steering_instructions.append({
                            "steering_vector": layer_vector,
                            "layer_idx": best_layer,
                            "coeff": adaptive_coeff,
                            "positions": steering_type
                        })
                        print(f"    ✓ {trait_name} @ layer {best_layer}: coeff {base_coeff:.2f} → {adaptive_coeff:.3f}")

            if not optimized_steering_instructions:
                print(f"No optimized steering instructions for {dataset}, skipping...")
                continue

            print(f"\nUsing {len(optimized_steering_instructions)} optimized steering vectors...")

            # Generate results using optimized steering
            prompt_builder = get_task_prompt_builder(dataset, task_name)

            print(f"Generating outputs with oracle preferences + adaptive steering...")
            steering_results = self.service.generate_batch_with_multiple_steering(
                texts=texts,
                task_prompt_builder=prompt_builder,
                steering_instructions=optimized_steering_instructions
            )

            # Create results with evaluation metrics
            rows = []
            for i, result in enumerate(steering_results):
                baseline_output = result.get("baseline_output", "")
                steered_output = result.get("steered_output", "")

                # Calculate evaluation metrics for steered output
                steered_metrics = self._compute_evaluation_metrics(
                    generated_text=steered_output,
                    reference_text=references[i] if i < len(references) else "",
                    preferences=preferences,
                    task_name=task_name
                )

                # Calculate evaluation metrics for baseline output
                baseline_metrics = self._compute_evaluation_metrics(
                    generated_text=baseline_output,
                    reference_text=references[i] if i < len(references) else "",
                    preferences=preferences,
                    task_name=task_name
                )

                row = {
                    "dataset": dataset,
                    "example_idx": i,
                    "input_text": texts[i],
                    "reference": references[i] if i < len(references) else "",
                    "baseline_output": baseline_output,
                    "oracle_adaptive_output": steered_output,
                    "preferences": "; ".join(preferences),
                    "num_steering_vectors": len(optimized_steering_instructions),
                    "approach": "oracle_preference_adaptive_steering"
                }

                # Add steered metrics with prefix
                for metric, value in steered_metrics.items():
                    row[f"oracle_adaptive_{metric}"] = value

                # Add baseline metrics with prefix
                for metric, value in baseline_metrics.items():
                    row[f"baseline_{metric}"] = value

                rows.append(row)

            # Save results
            df = pd.DataFrame(rows)
            safe_name = dataset.replace('/', '_').replace('-', '_')
            parquet_path = output_dir / f"{safe_name}_oracle_adaptive_results.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"Saved {len(rows)} results to {parquet_path}")

            results[dataset] = {
                'preferences': preferences,
                'num_examples': len(texts),
                'num_steering_vectors': len(optimized_steering_instructions),
                'output_file': str(parquet_path),
                'approach': 'oracle_preference_adaptive_steering'
            }

        # Push to HuggingFace if requested
        if hf_repo and results:
            try:
                url = push_parquet_results_to_hf(hf_repo, str(output_dir), repo_type='dataset')
                print(f"\nPushed oracle adaptive results to HuggingFace: {url}")
            except Exception as e:
                print(f"Failed to push to HuggingFace: {e}")

        return results

    def run_with_dynamic_inference(
        self,
        datasets: List[str],
        inferrer,  # HiddenStateSimilarityInferrer
        num_examples: int = 25,
        output_dir: str = "results_dynamic",
        hf_repo: Optional[str] = None,
        steering_layer: int = 20,
        steering_coeff: float = 2.0,
        steering_type: str = "response",
        mythos_data_dir: Optional[str] = None,
        csv_output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Run pipeline using dynamic inference (hidden state similarity) to get traits,
        then use those traits with the predefined steering system.
        """
        from prose.task_prompts import get_task_prompt_builder
        from utils.hf_upload import push_parquet_results_to_hf
        from pathlib import Path
        import pandas as pd

        print(f"=== Dynamic Inference Pipeline ===")
        print(f"Datasets: {datasets}")
        print(f"Inference method: Hidden State Similarity")
        print(f"Steering: layer={steering_layer}, coeff={steering_coeff}, type={steering_type}")

        # Use CSV loader if in CSV mode, otherwise use MLP loader
        if self.csv_mode:
            from data.csv_loader import CSVDatasetLoader, save_results_to_csv
            loader = CSVDatasetLoader()
        else:
            from data.mlp_loader import MLPDatasetLoader
            loader = MLPDatasetLoader(mythos_data_dir=mythos_data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        results = {}

        for dataset in datasets:
            print(f"\n=== Processing {dataset} with dynamic inference ===")

            # Load data
            items = loader.load_examples_with_metadata([dataset], num_examples)
            if not items:
                print(f"Skipping {dataset} - no data loaded")
                continue

            texts = [item['text'] for item in items]
            references = [item.get('reference', '') for item in items]
            task_name = items[0].get('task_name', 'summarization') if items else 'summarization'

            # Get true preferences for comparison
            true_preferences, _ = self.get_steering_vectors_for_dataset(dataset)
            print(f"True preferences for {dataset}: {true_preferences}")

            # Use hidden state similarity to infer traits with similarity scores
            print("Inferring traits using hidden state similarity...")

            if self.csv_mode:
                # For CSV mode, we'll infer traits per-example using their demo text
                # For now, use a representative demo text (from first item) for batch inference
                first_demo = items[0].get('user_demo', '') if items else ''
                if first_demo:
                    print(f"Using representative demo text for batch inference ({len(first_demo)} chars)")
                    # Use the inferrer's method that takes raw text
                    from prose.hidden_state_similarity import HiddenStateSimilarityInferrer
                    temp_inferrer = inferrer
                    # Extract hidden states from the demo text
                    hidden_states = temp_inferrer._extract_hidden_states(first_demo)
                    if hidden_states:
                        raw_similarities = temp_inferrer._compute_similarity_with_personas(hidden_states)
                        # Apply normalization
                        normalized_similarities = temp_inferrer._normalize_similarities_trait_baselines(raw_similarities)
                        # Filter traits
                        z_threshold = -0.5
                        min_raw_score = 0.05
                        filtered_traits = []
                        for trait, z_score in normalized_similarities.items():
                            raw_score = raw_similarities[trait]
                            if z_score >= z_threshold and raw_score >= min_raw_score:
                                filtered_traits.append((trait, z_score))

                        inferred_traits_with_scores = sorted(filtered_traits, key=lambda x: x[1], reverse=True)[:1]  # max_traits=1
                    else:
                        inferred_traits_with_scores = []
                else:
                    print("No demo text found in CSV items")
                    inferred_traits_with_scores = []
            else:
                # Regular dataset mode
                inferred_traits_with_scores = inferrer.infer_traits_with_scores_from_dataset(dataset, max_traits=1)

            print(f"Inferred traits with scores: {inferred_traits_with_scores}")

            if not inferred_traits_with_scores:
                print(f"No traitsw inferred for {dataset}, skipping steering...")
                continue

            # We need access to raw similarities for coefficient calculation
            # Get both z-scores and raw similarities
            print("Getting raw similarities for adaptive coefficients...")
            raw_similarities = inferrer._compute_similarity_with_personas(
                inferrer._extract_hidden_states(inferrer.dataset_demos[dataset])
            )

            # Create steering instructions using inferred traits with adaptive coefficients
            steering_instructions = []
            for trait, z_score in inferred_traits_with_scores:
                if trait in self.available_vectors:
                    try:
                        vector_path = self.available_vectors[trait]
                        vector = torch.load(vector_path, map_location='cpu')

                        # Handle 2D vectors [num_layers, hidden_size] by extracting the layer
                        if vector.ndim == 2:
                            if steering_layer < vector.shape[0]:
                                vector = vector[steering_layer]  # Extract specific layer
                            else:
                                vector = vector[-1]  # Use last layer if steering_layer too high
                        elif vector.ndim != 1:
                            print(f"  ✗ Unsupported vector shape {vector.shape} for {trait}")
                            continue

                        # Get raw similarity score for this trait
                        raw_similarity = raw_similarities.get(trait, 0.0)

                        # Calculate adaptive coefficient based on raw similarity
                        adaptive_coeff = inferrer.calculate_adaptive_coefficient(
                            raw_similarity,
                            base_coeff=steering_coeff,
                            similarity_threshold=inferrer.similarity_threshold,
                            trait_name=trait
                        )

                        steering_instructions.append({
                            "steering_vector": vector,
                            "layer_idx": steering_layer,
                            "coeff": adaptive_coeff,
                            "positions": steering_type
                        })
                        print(f"  ✓ Loaded {trait}: {vector.shape}, coeff: {adaptive_coeff:.3f} (raw_sim: {raw_similarity:.3f}, z-score: {z_score:.3f})")
                    except Exception as e:
                        print(f"  ✗ Failed to load {trait}: {e}")

            if not steering_instructions:
                trait_names = [trait for trait, _ in inferred_traits_with_scores]
                print(f"No steering vectors found for inferred traits: {trait_names}")
                continue

            print(f"Using {len(steering_instructions)} steering vectors...")

            # Generate results using the same pattern as run_with_predefined_vectors
            prompt_builder = get_task_prompt_builder(dataset, task_name)

            print(f"Generating outputs with dynamic steering...")
            steering_results = self.service.generate_batch_with_multiple_steering(
                texts=texts,
                task_prompt_builder=prompt_builder,
                steering_instructions=steering_instructions
            )

            # Create results dataframe
            rows = []
            for i, steering_result in enumerate(steering_results):
                trait_names = [trait for trait, _ in inferred_traits_with_scores]
                item_prefs = trait_names  # Use inferred traits as preferences
                human_answer = references[i] if i < len(references) else ""

                # Get both baseline and steered outputs from the result
                baseline_output = steering_result.get("baseline_output", "")
                steered_output = steering_result.get("steered_output", "")

                # Compute evaluation metrics
                eval_metrics = self._compute_evaluation_metrics(
                    generated_text=steered_output,
                    reference_text=human_answer,
                    preferences=true_preferences,  # Use true preferences as ground truth
                    task_name=task_name,
                    inferred_preferences=item_prefs  # Pass inferred traits for comparison
                )

                row = {
                    "dataset": dataset,
                    "query": steering_result.get("prompt", ""),
                    "example": texts[i] if i < len(texts) else "",
                    "human_answer": human_answer,
                    "baseline_output": baseline_output,
                    "personalized_answer": steered_output,
                    "true_preferences": "; ".join(true_preferences) if true_preferences else "",
                    "inferred_traits": "; ".join(trait_names) if trait_names else "",
                    "steering_layer": steering_layer,
                    "steering_coeff": steering_coeff,
                    "steering_type": steering_type,
                }

                # Add evaluation metrics
                row.update(eval_metrics)
                rows.append(row)

                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(texts)} examples...")

            # Save results
            df = pd.DataFrame(rows)
            safe_name = dataset.replace('/', '_')
            parquet_path = output_dir / f"{safe_name}_dynamic_results.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"Saved results: {parquet_path}")

            results[dataset] = {
                "traits": trait_names,
                "num_examples": len(rows),
                "file": str(parquet_path)
            }

        # Push to HuggingFace if requested
        if hf_repo:
            try:
                # Create a copy of all dataframes and normalize schema for HF upload
                output_dir_files = list(output_dir.glob("*.parquet"))
                for parquet_file in output_dir_files:
                    df_hf = pd.read_parquet(parquet_file)

                    # Normalize column names for schema compatibility
                    if 'true_preferences' in df_hf.columns:
                        df_hf = df_hf.rename(columns={'true_preferences': 'preferences'})

                    # Remove columns that cause schema conflicts
                    columns_to_remove = ['baseline_output', 'steering_layer', 'steering_coeff', 'steering_type']
                    for col in columns_to_remove:
                        if col in df_hf.columns:
                            df_hf = df_hf.drop(columns=[col])

                    # Ensure all new columns have default values for compatibility
                    if 'preference_similarity' not in df_hf.columns:
                        df_hf['preference_similarity'] = pd.NA

                    # Save the cleaned version
                    df_hf.to_parquet(parquet_file, index=False)

                url = push_parquet_results_to_hf(hf_repo, str(output_dir), repo_type='dataset')
                print(f"\nPushed dynamic results to HuggingFace: {url}")
            except Exception as e:
                print(f"Failed to push to HuggingFace: {e}")

        # Save CSV results if in CSV mode
        if self.csv_mode and csv_output_path:
            try:
                # Collect all results for CSV output
                all_csv_results = []
                for dataset, result_info in results.items():
                    parquet_file = result_info['file']
                    df = pd.read_parquet(parquet_file)

                    for _, row in df.iterrows():
                        csv_result = {
                            'csv_row_index': row.get('csv_row_index', ''),
                            'inferred_traits': row.get('inferred_traits', ''),
                            'generated_text': row.get('baseline_output', ''),
                            'steered_text': row.get('steered_output', ''),
                            'steering_layer': steering_layer,
                            'steering_coeff': steering_coeff,
                            'inference_method': 'hidden_state_similarity',
                            'evaluation_score': row.get('average_preference_match', ''),
                            'bertscore_f1': row.get('bertscore_f1', ''),
                            'preference_match': row.get('full_preference_match', '')
                        }
                        all_csv_results.append(csv_result)

                # Save to CSV using the first dataset (CSV file path)
                original_csv = datasets[0] if datasets else None
                if original_csv and all_csv_results:
                    save_results_to_csv(original_csv, all_csv_results, csv_output_path)
                    print(f"Saved CSV results to {csv_output_path}")

            except Exception as e:
                print(f"Failed to save CSV results: {e}")

        return results

    def rl(
        self,
        datasets: List[str],
        num_examples: int = 25,
        output_dir: str = "results_predefined",
        hf_repo: Optional[str] = None,
        steering_layer: int = 20,
        steering_coeff: float = 2.0,
        steering_type: str = "response",
        use_prompting: Optional[bool] = None
    ) -> Dict[str, any]:
        """
        Run pipeline using predefined steering vectors instead of inference.
        
        Args:
            use_prompting: If specified, overrides the instance setting for prompting.
                          If True, prompts will include trait descriptions to encourage 
                          the desired behaviors in addition to steering vectors.
        """
        from data.mlp_loader import MLPDatasetLoader
        from prose.task_prompts import get_task_prompt_builder
        from utils.hf_upload import push_parquet_results_to_hf
        import pandas as pd
        
        # Determine if we should use prompting for this run
        should_use_prompting = use_prompting if use_prompting is not None else self.use_prompting
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        loader = MLPDatasetLoader()
        
        for dataset in datasets:
            print(f"\n=== Processing {dataset} with predefined vectors ===")
            
            # Get predefined steering instructions for this dataset
            preferences, steering_instructions = self.get_steering_vectors_for_dataset(dataset)
            
            if not steering_instructions:
                print(f"Skipping {dataset} - no steering instructions available")
                continue
            
            # Load data
            items = loader.load_examples_with_metadata([dataset], num_examples)
            if not items:
                print(f"Skipping {dataset} - no data loaded")
                continue
            
            texts = [item['text'] for item in items]
            references = [item.get('reference', '') for item in items]
            task_name = items[0].get('task_name', 'summarization') if items else 'summarization'
            
            # Get actual trait names from steering instructions instead of layer numbers
            trait_names = []
            for inst in steering_instructions:
                # Find the vector name that corresponds to this steering instruction
                for pref in preferences:
                    vector_config = self.PREFERENCE_TO_VECTOR_MAP.get(pref)
                    if isinstance(vector_config, list):
                        for config in vector_config:
                            if config.get("layer") == inst["layer_idx"]:
                                trait_names.append(config["vector"])
                                break
                    elif isinstance(vector_config, dict):
                        if vector_config.get("layer") == inst["layer_idx"]:
                            trait_names.append(vector_config["vector"])
                    elif isinstance(vector_config, str):
                        trait_names.append(vector_config)
            
            # Remove duplicates while preserving order
            unique_trait_names = []
            for name in trait_names:
                if name not in unique_trait_names:
                    unique_trait_names.append(name)
            
            print(f"Using steering trait names: {unique_trait_names}")
            print(f"Corresponding to preferences: {preferences}")
            
            # Generate baseline (no steering) and steered outputs
            prompt_builder = get_task_prompt_builder(dataset, task_name)
            
            # Create modified prompt builder if prompting is enabled
            if should_use_prompting:
                trait_prompt = self._create_trait_prompting_text(preferences)
                print(f"Using trait prompting: {trait_prompt}")
                
                # Create a wrapper that adds trait prompting to the task prompt
                original_prompt_builder = prompt_builder
                def enhanced_prompt_builder(text):
                    base_prompt = original_prompt_builder(text)
                    if trait_prompt:
                        # Add trait prompting instruction to the end of the prompt
                        return f"{base_prompt}\n\n{trait_prompt}"
                    return base_prompt
                prompt_builder = enhanced_prompt_builder
            
            print(f"Generating outputs with multi-layer steering (prompting={'enabled' if should_use_prompting else 'disabled'})...")
            steering_results = self.service.generate_batch_with_multiple_steering(
                texts=texts,
                task_prompt_builder=prompt_builder,
                steering_instructions=steering_instructions
            )
            
            # Create results dataframe with evaluation metrics
            rows = []
            for i, steering_result in enumerate(steering_results):
                item_prefs = preferences  # Use predefined preferences
                human_answer = references[i] if i < len(references) else ""
                
                # Get both baseline and steered outputs from the same result
                baseline_output = steering_result.get("baseline_output", "")
                personalized_answer = steering_result.get("steered_output", "")
                
                # Add evaluation metrics (always compute to ensure consistent columns)
                eval_metrics = self._compute_evaluation_metrics(
                    generated_text=personalized_answer,
                    reference_text=human_answer,  # Can be empty
                    preferences=item_prefs,
                    task_name=task_name
                )
                
                row = {
                    "dataset": dataset,
                    "query": steering_result.get("prompt", ""),
                    "example": texts[i] if i < len(texts) else "",
                    "human_answer": human_answer,
                    "baseline_output": baseline_output,  # Add baseline output
                    "personalized_answer": personalized_answer,
                    "preferences": "; ".join(item_prefs) if item_prefs else "",
                    "inferred_traits": "; ".join(unique_trait_names) if unique_trait_names else "",
                    "prompting_enabled": should_use_prompting,
                    "trait_prompt": self._create_trait_prompting_text(item_prefs) if should_use_prompting else "",
                }
                
                # Add evaluation metrics to the row
                row.update(eval_metrics)
                rows.append(row)
            
            # Save results
            df = pd.DataFrame(rows)
            safe_name = dataset.replace('/', '_')
            parquet_path = output_dir / f"{safe_name}_predefined_results.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"Saved {len(rows)} results to {parquet_path}")
            
            results[dataset] = {
                'preferences': preferences,
                'steering_instructions': unique_trait_names,
                'num_examples': len(texts),
                'output_file': str(parquet_path)
            }
        
        # Push to HuggingFace if requested
        if hf_repo and results:
            try:
                from utils.hf_upload import push_parquet_results_to_hf
                url = push_parquet_results_to_hf(hf_repo, str(output_dir), repo_type='dataset')
                print(f"\nPushed results to HuggingFace: {url}")
            except Exception as e:
                print(f"Failed to push to HuggingFace: {e}")
        
        return results
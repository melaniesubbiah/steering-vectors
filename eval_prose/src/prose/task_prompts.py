"""Task prompt builders similar to PROSE for final generation tasks.

We avoid personalization here; just provide generic task prompts for datasets.
"""
from typing import Callable


def build_summary_prompt(text: str) -> str:
	return f"Please write a summary of the following discussion thread:\n\n{text}"


def build_email_prompt(notes: str) -> str:
	return f"Notes: {notes}\nPlease write an email based on the above notes:"


def get_task_prompt_builder(task_name: str = None) -> Callable[[str], str]:
	"""Get task prompt builder based on task type rather than just dataset name."""
	if task_name == 'email_writing':
		return build_email_prompt
	else:
		return build_summary_prompt 
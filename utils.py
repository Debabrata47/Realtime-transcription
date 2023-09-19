import re


def structure_summary(text: str):
    summaries = dict()

    # re pattern to categorize summaries
    speaker_summary_pattern = r'Summary of Speaker Analysis:\s(.*?)(?=\n\nKey Takeaways:)'
    key_takeaways_pattern = r'Key Takeaways:\s(.*?)(?=\n\Summary of Discussion Quality:)'
    quality_summary_pattern = r'Summary of Discussion Quality:\s(.*?)(?=\n\nSuggestions for Improvement:)'
    improvement_summary_pattern = r'Suggestions for Improvement:(.*)'

    # Extracting the content based on re
    speaker_summary_match = re.search(speaker_summary_pattern, text, re.DOTALL)
    key_takeaways_match = re.search(key_takeaways_pattern, text, re.DOTALL)
    quality_summary_match = re.search(quality_summary_pattern, text, re.DOTALL)
    improvement_summary_match = re.search(improvement_summary_pattern, text, re.DOTALL)

    if speaker_summary_match:
        topic = speaker_summary_match.group(1).strip()
        summaries['Summary of Speaker Analysis'] = topic

    if key_takeaways_match:
        topic = key_takeaways_match.group(1).strip()
        summaries['Key Takeaways'] = topic

    if quality_summary_match:
        topic = quality_summary_match.group(1).strip()
        summaries['Summary of Discussion Quality'] = topic

    if improvement_summary_match:
        topic = improvement_summary_match.group(1).strip()
        summaries['Suggestions for Improvement'] = topic

    return summaries

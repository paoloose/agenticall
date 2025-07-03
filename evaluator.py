import json
import os
import statistics
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from poc import OPENAI_API_KEY

TRANSCRIPTS_DIR = "transcripts"
REPORTS_DIR = "reports"

# Performance evaluation prompt with Chain-of-Thought reasoning


PERFORMANCE_EVALUATION_PROMPT = """
You are a performance evaluator for McRouter customer support calls. You need to analyze the following call transcript and evaluate 5 key performance indicators (KPIs) using Chain-of-Thought reasoning.

**KPI Definitions:**
1. **First Call Resolution (FCR)** - Score 0-100: Percentage indicating if the problem was resolved in the first call without need for follow-up
2. **Average Handle Time (AHT)** - Score 0-100: Efficiency score based on call duration (shorter appropriate duration = higher score)
3. **Frontline Forward Rate (FRR)** - Score 0-100: Penalty score for using forwards (no forwards = 100, multiple forwards = lower score)
4. **Average Speed of Answer (ASA)** - Score 0-100: How quickly the system responded to the customer (immediate = 100)
5. **Forward Context Sufficiency (FCS)** - Score 0-100: Quality and sufficiency of forward context for human agents (detailed context = higher score)

**Call Transcript:**
{transcript}

**Instructions:**
For each KPI, follow this Chain-of-Thought process:
1. **Analysis**: Examine the transcript for relevant indicators
2. **Evidence**: Identify specific evidence from the transcript
3. **Reasoning**: Explain your scoring logic step by step
4. **Score**: Assign a numerical score (0-100)

**Output Format:**
Provide your analysis in the following JSON format:
{{
  "FCR": {{"score": <number>, "reasoning": "<detailed reasoning>"}},
  "AHT": {{"score": <number>, "reasoning": "<detailed reasoning>"}},
  "FRR": {{"score": <number>, "reasoning": "<detailed reasoning>"}},
  "ASA": {{"score": <number>, "reasoning": "<detailed reasoning>"}},
  "FCS": {{"score": <number>, "reasoning": "<detailed reasoning>"}}
}}

Think step by step and provide comprehensive reasoning for each score.

You must ONLY output raw JSON message starting with an open curly brace:"""

class PerformanceEvaluator:
  def __init__(self):
    self.evaluator_model = ChatOpenAI(
      model="gpt-4o",
      temperature=0.1,
      api_key=OPENAI_API_KEY,
    )
    self.prompt_template = PromptTemplate.from_template(PERFORMANCE_EVALUATION_PROMPT)

  def load_transcript(self, filepath: str) -> Dict[str, Any]:
    """Load a transcript from a JSON file"""
    try:
      with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    except Exception as e:
      print(f"Error loading transcript {filepath}: {e}")
      return None

  def format_transcript_for_evaluation(self, transcript: Dict[str, Any]) -> str:
      """
      Format transcript data for better evaluation
      """
      formatted = f"Call ID: {transcript.get('id', 'Unknown')}\n"
      formatted += f"Date: {transcript.get('date', 'Unknown')}\n"
      formatted += f"Duration: {transcript.get('duration', 0)} seconds\n\n"
      formatted += "Conversation Flow:\n"

      for msg in transcript.get('messages', []):
        if msg['type'] == 'message':
          role = "Agent" if msg['role'] == 'system' else "Customer"
          formatted += f"[{msg['time']}s] {role}: {msg['message']}\n"
        elif msg['type'] == 'function_call':
          formatted += f"[{msg['time']}s] Function Call: {msg['function']}\n"
          if 'params' in msg:
            formatted += f"  Parameters: {json.dumps(msg['params'], indent=2)}\n"

      return formatted

  def evaluate_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single transcript and return KPI scores."""
    try:
      # Format transcript for evaluation
      formatted_transcript = self.format_transcript_for_evaluation(transcript)

      # Create the evaluation prompt
      evaluation_prompt = self.prompt_template.format(transcript=formatted_transcript)

      # Get evaluation from the model
      response = self.evaluator_model.invoke([
        SystemMessage("You are an expert customer service performance evaluator. Analyze transcripts thoroughly and provide accurate KPI scores with detailed reasoning."),
        HumanMessage(evaluation_prompt)
      ])

      # We clean the response for markdown-like outputs
      content = response.content.replace("```", "")
      content = content.replace("json\n", "")
      try:
        evaluation_result = json.loads(content)

        print(evaluation_result)

        # Add metadata
        evaluation_result["metadata"] = {
          "call_id": transcript.get("id", "Unknown"),
          "date": transcript.get("date", "Unknown"),
          "duration": transcript.get("duration", 0),
          "evaluated_at": datetime.now().isoformat()
        }

        return evaluation_result

      except json.JSONDecodeError as e:
        print(f"Error parsing evaluation response: {e}")
        print(f"Response content: {content}")
        return None

    except Exception as e:
      print(f"Error evaluating transcript: {e}")
      return None

  def evaluate_all_transcripts(self) -> List[Dict[str, Any]]:
    """Evaluate all transcripts in the transcripts directory."""
    evaluations = []

    if not os.path.exists(TRANSCRIPTS_DIR):
      print(f"Transcripts directory '{TRANSCRIPTS_DIR}' not found!")
      return evaluations

    transcript_files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith('.json')]

    if not transcript_files:
      print("No transcript files found!")
      return evaluations

    print(f"Found {len(transcript_files)} transcript files to evaluate...")

    for filename in transcript_files:
      filepath = os.path.join(TRANSCRIPTS_DIR, filename)
      print(f"Evaluating {filename}...")

      transcript = self.load_transcript(filepath)
      if transcript:
        evaluation = self.evaluate_transcript(transcript)
        if evaluation:
          evaluations.append(evaluation)
          print(f"- Successfully evaluated {filename}")
        else:
          print(f"- Failed to evaluate {filename}")
      else:
        print(f"- Failed to load {filename}")

    return evaluations

  def generate_performance_report(self, evaluations: List[Dict[str, Any]]):
    if not evaluations:
      print("No evaluations to generate report from!")
      return

    if not os.path.exists(REPORTS_DIR):
      os.makedirs(REPORTS_DIR)

    # KPIs definidos por nosotros :)
    kpis = ['FCR', 'AHT', 'FRR', 'ASA', 'FCS']
    kpi_data = {kpi: [] for kpi in kpis}
    call_ids = []

    for evaluation in evaluations:
      call_ids.append(evaluation['metadata']['call_id'])
      for kpi in kpis:
        if kpi in evaluation and 'score' in evaluation[kpi]:
          kpi_data[kpi].append(evaluation[kpi]['score'])
        else:
          kpi_data[kpi].append(0) # default

    # Comprehensive report dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
      'McRouter Customer Support Performance Dashboard',
      fontsize=24,
      fontweight='bold',
      y=0.95,
    )

    # Color theme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # 1. KPI Overview Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    kpi_averages = [statistics.mean(kpi_data[kpi]) for kpi in kpis]
    bars = ax1.bar(kpis, kpi_averages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Average KPI Scores', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Score (0-100)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, avg in zip(bars, kpi_averages):
      height = bar.get_height()
      ax1.text(
        bar.get_x() + bar.get_width()/2., height + 1,
        f'{avg:.1f}', ha='center', va='bottom', fontweight='bold',
      )

    # 2. KPI Distribution Box Plot
    ax2 = fig.add_subplot(gs[1, 0])
    bp = ax2.boxplot([kpi_data[kpi] for kpi in kpis], tick_labels=kpis, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
      patch.set_facecolor(color)
      patch.set_alpha(0.7)
    ax2.set_title('KPI Score Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score (0-100)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Performance Trend (if multiple calls)
    ax3 = fig.add_subplot(gs[1, 1])
    if len(evaluations) > 1:
      for i, kpi in enumerate(kpis):
        ax3.plot(
          range(len(kpi_data[kpi])), kpi_data[kpi],
          marker='o', linewidth=2, label=kpi, color=colors[i],
        )
      ax3.set_title('Performance Trends Across Calls', fontsize=14, fontweight='bold')
      ax3.set_xlabel('Call Number', fontsize=12)
      ax3.set_ylabel('Score (0-100)', fontsize=12)
      ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
      ax3.grid(True, alpha=0.3)
    else:
      ax3.text(
        0.5, 0.5, 'Need multiple calls\nfor trend analysis',
        ha='center', va='center', transform=ax3.transAxes, fontsize=12,
      )
      ax3.set_title('Performance Trends', fontsize=14, fontweight='bold')

    # 4. Performance Radar Chart
    ax4 = fig.add_subplot(gs[1, 2], projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(kpis), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    avg_scores = kpi_averages + [kpi_averages[0]]  # Complete the circle
    ax4.plot(angles, avg_scores, 'o-', linewidth=2, color='#FF6B6B', alpha=0.8)
    ax4.fill(angles, avg_scores, alpha=0.25, color='#FF6B6B')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(kpis)
    ax4.set_ylim(0, 100)
    ax4.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True)

    # 5. Call Duration vs Performance
    ax5 = fig.add_subplot(gs[2, 0])
    durations = [eval['metadata']['duration'] for eval in evaluations]
    overall_scores = [statistics.mean([eval[kpi]['score'] for kpi in kpis]) for eval in evaluations]
    scatter = ax5.scatter(durations, overall_scores, c=overall_scores,
                        cmap='RdYlGn', s=100, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Call Duration (seconds)', fontsize=12)
    ax5.set_ylabel('Overall Performance Score', fontsize=12)
    ax5.set_title('Duration vs Performance', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Performance Score')

    # 6. Forward Rate Analysis
    ax6 = fig.add_subplot(gs[2, 1])
    forward_counts = []
    for evaluation in evaluations:
      # Count forward calls in the original transcript
      call_id = evaluation['metadata']['call_id']
      forward_count = 0
      # This would need to be extracted from the original transcript
      # For now, we'll use the FRR score as an inverse indicator
      frr_score = evaluation['FRR']['score']
      # Estimate forwards: higher FRR score = fewer forwards
      estimated_forwards = max(0, int((100 - frr_score) / 20))
      forward_counts.append(estimated_forwards)

    forward_hist = ax6.hist(forward_counts, bins=range(max(forward_counts) + 2),
                            alpha=0.7, color='#4ECDC4', edgecolor='black')
    ax6.set_xlabel('Number of Forwards', fontsize=12)
    ax6.set_ylabel('Number of Calls', fontsize=12)
    ax6.set_title('Forward Frequency Distribution', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Performance Summary Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')

    # Create summary statistics
    summary_data = []
    for kpi in kpis:
      scores = kpi_data[kpi]
      summary_data.append([
        kpi,
        f"{statistics.mean(scores):.1f}",
        f"{min(scores):.1f}",
        f"{max(scores):.1f}",
        f"{statistics.stdev(scores):.1f}" if len(scores) > 1 else "0.0"
      ])

    table = ax7.table(
      cellText=summary_data,
      colLabels=['KPI', 'Average', 'Min', 'Max', 'Std Dev'],
      cellLoc='center',
      loc='center',
      colWidths=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(kpis)):
      table[(i+1, 0)].set_facecolor(colors[i])
      table[(i+1, 0)].set_alpha(0.3)

    ax7.set_title('Performance Statistics Summary', fontsize=14, fontweight='bold')

    # 8. Recent Performance Trend
    ax8 = fig.add_subplot(gs[3, :])
    if len(evaluations) > 1:
      # Sort evaluations by date
      sorted_evals = sorted(evaluations, key=lambda x: x['metadata']['date'])
      dates = [eval['metadata']['date'] for eval in sorted_evals]

      # Calculate overall scores for each call
      overall_scores = []
      for eval in sorted_evals:
        scores = [eval[kpi]['score'] for kpi in kpis]
        overall_scores.append(statistics.mean(scores))

      ax8.plot(range(len(overall_scores)), overall_scores,
              marker='o', linewidth=3, markersize=8, color='#FF6B6B', alpha=0.8)
      ax8.set_xlabel('Call Sequence', fontsize=12)
      ax8.set_ylabel('Overall Performance Score', fontsize=12)
      ax8.set_title('Overall Performance Trend Over Time', fontsize=14, fontweight='bold')
      ax8.grid(True, alpha=0.3)
      ax8.set_ylim(0, 100)

      # Add trend line
      if len(overall_scores) > 2:
        z = np.polyfit(range(len(overall_scores)), overall_scores, 1)
        p = np.poly1d(z)
        ax8.plot(
          range(len(overall_scores)), p(range(len(overall_scores))),
          "--", alpha=0.7, color='blue', linewidth=2, label='Trend',
        )
        ax8.legend()
    else:
      ax8.text(
        0.5, 0.5, 'Need multiple calls for trend analysis',
        ha='center', va='center', transform=ax8.transAxes, fontsize=14,
      )
      ax8.set_title('Overall Performance Trend', fontsize=14, fontweight='bold')

    # Add footer with summary
    footer_text = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    footer_text += f"Total Calls Analyzed: {len(evaluations)} | "
    footer_text += f"Overall Average Score: {statistics.mean([statistics.mean([eval[kpi]['score'] for kpi in kpis]) for eval in evaluations]):.1f}/100"

    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10, style='italic')

    # Save the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"performance_report_{timestamp}.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance report saved to: {report_path}")

    # Also save detailed evaluation results
    results_path = os.path.join(REPORTS_DIR, f"detailed_evaluations_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
      json.dump(evaluations, f, indent=2, ensure_ascii=False)
    print(f"📋 Detailed evaluations saved to: {results_path}")

    plt.show()

    # Print summary to console
    print("\n" + "="*40)
    print("Resumen de performance:")
    print("="*40)
    print(f"Total Calls Analyzed: {len(evaluations)}")
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nKPI Averages:")
    for kpi, avg in zip(kpis, kpi_averages):
      print(f"  {kpi}: {avg:.1f}/100")

    overall_avg = statistics.mean(kpi_averages)
    print(f"\nOverall Average Performance: {overall_avg:.1f}/100")

def main():
  print("McRouter Performance Evaluator")
  print("--------------------------------")

  evaluator = PerformanceEvaluator()

  print("Comenzando la evaluación de transcripts...")
  evaluations = evaluator.evaluate_all_transcripts()

  if evaluations:
    print("Generando reporte de performance...")
    evaluator.generate_performance_report(evaluations)
  else:
    print("No hay transcripciones. Reporte no generado.")

main()

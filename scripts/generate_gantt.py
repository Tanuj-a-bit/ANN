import matplotlib.pyplot as plt
import pandas as pd
import os

# Professional dark palette
BG_COLOR = "#0D1117"
BAR_COLOR = "#58A6FF"
TEXT_COLOR = "#C9D1D9"
EDGE_COLOR = "#30363D"
GRID_COLOR = "#30363D"

def draw_gantt():
    os.makedirs('assets', exist_ok=True)
    
    tasks = [
        "Topic Selection & Synopsis Preparation",
        "Literature Study & Requirement Analysis",
        "Data Collection & Preprocessing",
        "System Design & Model Development",
        "Testing & Performance Evaluation",
        "Documentation & Final Review"
    ]
    
    # Define start and duration based on the table provided (1-indexed weeks)
    # Topic: W1-2 (start=0, dur=2)
    # Lit: W2-4 (start=1, dur=3)
    # Data: W4-5 (start=3, dur=2)
    # System: W5-8 (start=4, dur=4)
    # Testing: W7-9 (start=6, dur=3)
    # Doc: W10-12 (start=9, dur=3)
    
    start_weeks = [0, 1, 3, 4, 6, 9]
    durations = [2, 3, 2, 4, 3, 3]
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Reversed to show first task at top
    for i, (task, start, dur) in enumerate(zip(tasks[::-1], start_weeks[::-1], durations[::-1])):
        ax.barh(task, dur, left=start, color=BAR_COLOR, edgecolor=EDGE_COLOR, height=0.6, alpha=0.85)
        # Add labels identifying weeks
        ax.text(start + dur/2, task, f'W{start+1}-W{start+dur}', color='#FFFFFF', ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Project Timeline (Weeks)', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax.set_title('Detailed Project Implementation Roadmap', color=TEXT_COLOR, fontsize=18, fontweight='bold', pad=30)
    
    # Set x-ticks for weeks 1 to 12
    ax.set_xticks(range(0, 13))
    ax.set_xticklabels([f'W{i}' for i in range(1, 14)])
    
    # Grid and Spines
    ax.xaxis.grid(True, linestyle='--', alpha=0.15, color=GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(EDGE_COLOR)
    ax.spines['bottom'].set_color(EDGE_COLOR)
    
    plt.tight_layout()
    plt.savefig('assets/project_timeline_gantt.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print("Realistic Gantt Chart generated successfully.")

if __name__ == "__main__":
    draw_gantt()

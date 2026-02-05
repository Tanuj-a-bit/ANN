import matplotlib.pyplot as plt
import os

# Academic theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def draw_academic_gantt():
    os.makedirs('assets', exist_ok=True)
    
    tasks = [
        "Topic Selection & Synopsis Preparation",
        "Literature Study & Requirement Analysis",
        "Data Collection & Preprocessing",
        "System Design & Model Development",
        "Testing & Performance Evaluation",
        "Documentation & Final Review"
    ]
    
    # Start weeks (0-indexed) and Durations (based on user ranges)
    # W1-2 -> Start: 0, Duration: 2
    # W2-4 -> Start: 1, Duration: 3
    # W4-5 -> Start: 3, Duration: 2
    # W5-8 -> Start: 4, Duration: 4
    # W7-9 -> Start: 6, Duration: 3
    # W10-12 -> Start: 9, Duration: 3
    
    starts = [0, 1, 3, 4, 6, 9]
    durations = [2, 3, 2, 4, 3, 3]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a professional blue color
    BAR_COLOR = '#2B579A' # Royal Blue (Academic Style)
    
    # Reverse tasks to have the first task at the top
    y_pos = range(len(tasks))
    tasks_rev = tasks[::-1]
    starts_rev = starts[::-1]
    durs_rev = durations[::-1]
    
    bars = ax.barh(y_pos, durs_rev, left=starts_rev, color=BAR_COLOR, height=0.6, edgecolor='none')
    
    # Label weeks 1 to 12
    ax.set_xticks(range(0, 13))
    ax.set_xticklabels([f'Week {i}' for i in range(1, 14)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks_rev, fontsize=12)
    
    # Style the grid and spines
    ax.set_xlim(0, 12)
    ax.grid(axis='x', linestyle='-', alpha=0.2, color='gray')
    ax.xaxis.tick_top() # Put labels at top
    ax.xaxis.set_label_position('top')
    
    # Remove unnecessary spines
    for spine in ['right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Adjust layout
    plt.title("Engineering Project Implementation Timeline", fontsize=18, fontweight='bold', pad=40)
    plt.tight_layout()
    
    # Save high resolution image
    plt.savefig('assets/academic_gantt.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Academic Gantt Chart saved as assets/academic_gantt.png")

if __name__ == "__main__":
    draw_academic_gantt()

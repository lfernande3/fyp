# Visual Assets Guide for PowerPoint Presentation

## Essential Figures to Generate from Your Code

### 1. Main Trade-off Scatter Plot (Slide 14)
```python
# From objective_o2_demo.ipynb
# Shows lifetime vs. delay for different ts values
# Make sure to:
- Use large, clear markers
- Add trend lines
- Label each ts curve clearly
- Highlight the Pareto frontier
- Use colorblind-friendly palette
```

### 2. Transmission Probability Analysis (Slide 15)
```python
# Two subplots: lifetime vs. q and delay vs. q
# Key elements:
- Show the monotonic decrease in lifetime
- Highlight the U-shaped delay curve
- Mark the optimal q = 1/n point
- Add shaded regions for "good" operating zones
```

### 3. Energy Consumption Breakdown (Slide 10)
```python
# Bar chart or pie chart showing power by state
# Suggestions:
- Use log scale to show PS = 0.01mW vs PT = 100mW
- Add icons for each state
- Use color coding: red for high power, green for low
```

### 4. State Evolution Timeline (Slide 9)
```python
# Time series showing state transitions
# Include:
- Color bands for each state
- Annotations for key events (packet arrival, transmission)
- Show ~100 time slots for clarity
```

### 5. Queue Length Dynamics (Slide 18 - Demo)
```python
# Interactive plot showing queue evolution
# Features:
- Real-time updates as parameters change
- Highlight congestion periods
- Show average queue length line
```

## Design Templates for Diagrams

### 1. System Architecture (Slide 8)
```
PowerPoint SmartArt suggestions:
- Use "Process" > "Basic Process" for flow
- Color code: Blue for input, Green for processing, Orange for output
- Add icons from Insert > Icons:
  - Gear icon for Configuration
  - Network icon for Simulator
  - Chart icon for Results
```

### 2. Node State Machine (Slide 9)
```
PowerPoint shapes:
- Rounded rectangles for states
- Curved arrows for transitions
- Color scheme:
  - SLEEP: Dark blue (low energy)
  - IDLE: Light blue
  - ACTIVE: Green (productive)
  - WAKEUP: Yellow (transitioning)
- Add timing annotations (ts, tw) on arrows
```

### 3. IoT Ecosystem (Slide 3)
```
Visual elements:
- Central cloud/network hub
- Radiating connections to devices:
  - Smart meter icon
  - Temperature sensor icon
  - Parking sensor icon
  - Medical device icon
- Use perspective to show scale
```

### 4. Timeline Gantt Chart (Slide 7)
```
Show project progress:
- O1: [████████████] 100% ✓
- O2: [████████████] 100% ✓
- O3: [████--------] 40%
- O4: [██----------] 15%
- Use green for completed, blue for in-progress
```

## Color Palette Recommendations

### Primary Palette
```
- Primary Blue: #2E86AB (headers, main elements)
- Accent Orange: #F24236 (highlights, important data)
- Success Green: #2ECC71 (positive results)
- Neutral Gray: #7F8C8D (supporting text)
- Background: #FFFFFF or #F8F9FA
```

### For Plots (Colorblind-Friendly)
```
- Series 1: #0173B2 (blue)
- Series 2: #DE8F05 (orange)
- Series 3: #029E73 (green)
- Series 4: #CC78BC (purple)
- Series 5: #CA9161 (brown)
```

## Icon Resources

### Free Icon Sources
1. **PowerPoint Built-in Icons** (Insert > Icons)
   - Search for: IoT, sensor, battery, network, time
   
2. **Recommended Icons to Include:**
   - Battery with percentage for energy
   - Clock for latency/delay
   - Network nodes for devices
   - Lightning bolt for transmission
   - Moon for sleep state
   - Graph for analytics

### Creating Custom Graphics

#### Battery Life Visualization
```
Create in PowerPoint:
1. Insert rectangle (battery outline)
2. Add smaller rectangles (charge level)
3. Use gradient fill:
   - 100%: Green
   - 50%: Yellow
   - 20%: Red
4. Add percentage text
5. Group and save as image
```

#### Latency Clock
```
1. Insert circle
2. Add clock hands
3. Use motion blur effect
4. Color code:
   - Green: <10ms (low latency)
   - Yellow: 10-100ms (medium)
   - Red: >100ms (high)
```

## Animation Suggestions

### Slide Transitions
- Use subtle transitions: Fade, Push, or Morph
- Duration: 0.5-1 second max
- Avoid distracting effects

### On-Slide Animations
1. **State Machine (Slide 9)**
   - Animate packet moving through states
   - Highlight active state with glow effect
   - Show energy bar depleting

2. **Results Reveal (Slides 14-15)**
   - Data points appear in sequence
   - Draw trend line after points
   - Highlight key findings with zoom

3. **Architecture Flow (Slide 8)**
   - Sequential appearance of components
   - Animated arrows showing data flow
   - Pulse effect on active component

## Screenshot Preparation

### For Demo Slide (18)
```python
# Capture screenshots showing:
1. Parameter sliders interface
2. Real-time plot updates
3. Numerical results table
4. State distribution pie chart

# Use high DPI:
plt.figure(dpi=150)
# Save with transparency:
plt.savefig('demo_screenshot.png', transparent=True, bbox_inches='tight')
```

### Code Snippets
```python
# Format code for slides:
- Use syntax highlighting
- Keep to 5-10 lines max
- Focus on key algorithms
- Use large font (14pt+)
```

## Infographic Ideas

### Slide 4: IoT Growth
```
2020: ●●●●● 10B devices
2025: ●●●●●●●●●●●●●●● 30B devices
2030: ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 75B devices
(Each ● = 2B devices)
```

### Slide 16: Scenario Comparison
```
Low Latency Mode        Battery Life Mode
     📱                      📱
   ↯↯↯↯↯                    ↯
[█░░░░] 2.3 yrs        [████████] 11.5 yrs
⏱ 8.2 slots            ⏱ 41.7 slots
```

### Slide 20: Decision Tree
```
                Application Type?
                /              \
        Time-Critical      Battery-Critical
              |                    |
         ts < 10              ts > 50
         q > 0.1              q < 0.05
              |                    |
      [Emergency Alert]    [Environmental Sensor]
```

## Best Practices

### Do's
- ✓ High contrast between text and background
- ✓ Consistent font sizes (Title: 40pt, Body: 24pt)
- ✓ White space for breathing room
- ✓ Align elements to grid
- ✓ Test on projector beforehand

### Don'ts
- ✗ Cluttered slides with too much info
- ✗ Small fonts (<20pt)
- ✗ Low quality/pixelated images
- ✗ Excessive animations
- ✗ Reading directly from slides

## Quick PowerPoint Tips

### Master Slide Setup
1. View > Slide Master
2. Set consistent fonts and colors
3. Add subtle footer with slide numbers
4. Include your institution logo

### Alignment Tools
- Home > Arrange > Align
- Use guides (View > Guides)
- Snap to grid for consistency

### Export Settings
- File > Export > Create PDF
- Use "Standard" quality for sharing
- Keep PPT for presentation, PDF for handouts

## Final Checklist

Before presentation:
- [ ] All plots generated at high resolution
- [ ] Color scheme consistent throughout
- [ ] Animations tested and timed
- [ ] Backup slides prepared
- [ ] PDF version exported
- [ ] Demo environment ready
- [ ] Clicker/pointer tested
- [ ] Timer visible for pacing
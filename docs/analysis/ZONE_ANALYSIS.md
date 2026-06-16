"""
PITCH ZONE ANALYSIS: Shot-in-10s Rate by Defensive Action Location

The pitch is divided into a 12×8 grid (12 columns, 8 rows)
- Columns (0-12): Attacking direction LEFT→RIGHT
- Rows (0-8): Vertical pitch position

ZONE MAPPING:
- Left side (x: 0-4) = DEFENSIVE THIRD (your team's goal)
- Center (x: 4-8) = MIDDLE THIRD
- Right side (x: 8-12) = ATTACKING THIRD (opponent's goal)

- Bottom rows (y: 0-3) = LEFT FLANK
- Center rows (y: 3-5) = CENTER
- Top rows (y: 5-8) = RIGHT FLANK
"""

print("=" * 80)
print("ZONE-BY-ZONE BREAKDOWN: Shot-in-10s Rate by Defensive Action Location")
print("=" * 80)

# Define zones with shot rates from the heatmap
zones = {
    # DEFENSIVE THIRD (Left side - x: 0-4)
    "Zone 0-1 (Far left, center)": {
        "location": "Your own goal area, center",
        "shot_rate": 0.03,
        "context": "Defensive clearance in your own penalty area",
        "example": "Liverpool defender clears from their own 6-yard box (center)"
    },
    "Zone 1-2 (Left flank, lower)": {
        "location": "Your own goal area, left wing",
        "shot_rate": 0.06,
        "context": "Defensive action on left flank near own goal",
        "example": "Manchester City left-back clears from their own corner area"
    },
    "Zone 3-4 (Left flank, upper)": {
        "location": "Your own goal area, right wing",
        "shot_rate": 0.09,
        "context": "Defensive action on right wing near own goal",
        "example": "Arsenal right-back intercepts on the wing near their goal"
    },

    # MIDDLE THIRD - TRANSITION ZONE (x: 4-8)
    "Zone 4-6 (Center midfield)": {
        "location": "Midfield center zone",
        "shot_rate": 0.03-0.05,
        "context": "Midfield contest - neither team close to goal",
        "example": "Chelsea midfielder wins ball in center field. Rarely converts immediately"
    },
    "Zone 5-7 (Midfield flanks)": {
        "location": "Midfield left/right wing",
        "shot_rate": 0.04-0.09,
        "context": "Wing play in midfield",
        "example": "Liverpool wins possession on midfield flank. 4-9% chance of shot"
    },

    # ATTACKING THIRD (Right side - x: 8-12) ⚠️ HIGHEST DANGER
    "Zone 9-10 (Attacking 3rd left flank, LOWER)": {
        "location": "Attacking left wing, inside opponent's defensive third",
        "shot_rate": 0.14,
        "context": "⚠️ HIGH DANGER ZONE - Wide attacking area",
        "example": "Man City left winger wins ball in opponent's box, left side. 14% chance of immediate shot"
    },

    "Zone 10-11 (Attacking 3rd CENTER)": {
        "location": "Central attacking area, just outside opponent's penalty box",
        "shot_rate": 0.16-0.19,
        "context": "🔴 CRITICAL DANGER ZONE - Best shooting position",
        "example": "Real Madrid midfielder wins contested ball 25 yards out, central. 16-19% chance of immediate shot"
    },

    "Zone 11-12 (Attacking 3rd RIGHT FLANK, UPPER)": {
        "location": "Attacking right wing, but further from box",
        "shot_rate": 0.13,
        "context": "HIGH DANGER - Wide attacking opportunity",
        "example": "Barcelona winger regains possession on right wing in attacking area. 13% shot chance"
    },

    # SPECIAL ZONES - EXTREME FLANKS
    "Zone 2-3 (VERY LEFT FLANK - near own goal)": {
        "location": "Far left wing, your defensive area",
        "shot_rate": 0.09-0.10,
        "context": "Wing clearance near own goal",
        "example": "Liverpool left-back clears from corner. 9-10% shot chance (surprisingly high!)"
    },

    "Zone 10-11 (VERY RIGHT FLANK - attacking goal)": {
        "location": "Far right wing, opponent's defensive area",
        "shot_rate": 0.17-0.19,
        "context": "🔴 EXTREME DANGER - Combination play on wing",
        "example": "Napoli winger combines with fullback on right wing near goal. 17-19% shot probability"
    }
}

print("\n" + "=" * 80)
print("RANK BY DANGER LEVEL (Shot-in-10s Rate)")
print("=" * 80)

danger_levels = [
    ("🔴 ULTRA-DANGER", "Zone 10-11 (Center attacking 3rd)", 0.19, "19% shot rate - Immediate shooting positions"),
    ("🔴 VERY HIGH", "Zone 10-11 (Center attacking 3rd)", 0.16, "16-19% shot rate - Best attacking locations"),
    ("🟠 HIGH", "Zone 9-10 + 11-12 (Flanks attacking 3rd)", 0.13-0.14, "13-14% shot rate - Wide attacking play"),
    ("🟡 MODERATE", "Zone 2-3 + 9-10 (Attacking flanks)", 0.09-0.10, "9-10% shot rate - Wing clearances & transitions"),
    ("🟢 LOW", "Zone 4-8 (Midfield)", 0.03-0.05, "3-5% shot rate - Contested midfield"),
    ("🟢 VERY LOW", "Zone 0-1 (Deep defense)", 0.03, "3% shot rate - Defensive clearances from own goal")
]

for danger, zones_list, rate, meaning in danger_levels:
    print(f"\n{danger}: {rate}")
    print(f"  Zones: {zones_list}")
    print(f"  Meaning: {meaning}")

print("\n" + "=" * 80)
print("PRACTICAL EXAMPLES BY CLUB & ZONES")
print("=" * 80)

examples = [
    {
        "club": "Liverpool vs Man City",
        "scenario": "City wins possession on left wing, in Man City's attacking third",
        "zones": "Zone 9-10 (left flank, attacking)",
        "shot_rate": 0.14,
        "interpretation": "Man City has 14% chance of an immediate shot. This is high! Why? Because they're on the wing DEEP in opponent territory with fresh attacking momentum."
    },
    {
        "club": "Real Madrid vs Barcelona",
        "scenario": "Madrid midfielder intercepts in central attacking area",
        "zones": "Zone 10-11 (center, attacking 3rd)",
        "shot_rate": 0.19,
        "interpretation": "Madrid has 19% chance of immediate shot. HIGHEST PROBABILITY. They're central, close to goal, with possession."
    },
    {
        "club": "Bayern Munich vs Dortmund",
        "scenario": "Dortmund defender clears from own penalty box",
        "zones": "Zone 0-1 (center, defensive 3rd)",
        "shot_rate": 0.03,
        "interpretation": "Only 3% Bayern chance of immediate shot. Dortmund just cleared desperately - far from goal, no setup time."
    },
    {
        "club": "Arsenal vs Tottenham",
        "scenario": "Arsenal wins ball in Tottenham penalty area, left side",
        "zones": "Zone 9-10 lower (left flank, attacking 3rd)",
        "shot_rate": 0.14,
        "interpretation": "14% shot chance. Arsenal wide attacking opportunity - not quite central, but in dangerous territory."
    },
    {
        "club": "Chelsea vs Newcastle",
        "scenario": "Chelsea midfielder wins contested ball in center midfield",
        "zones": "Zone 6-7 (center midfield)",
        "shot_rate": 0.04,
        "interpretation": "Only 4% shot rate. Too far from goal, opponent has time to reorganize defense."
    },
    {
        "club": "PSG vs Marseille",
        "scenario": "PSG fullback wins ball on right wing, recovering it 30 yards from goal",
        "zones": "Zone 10-11 (right flank, attacking 3rd)",
        "shot_rate": 0.13,
        "interpretation": "13% shot chance. High-value possession recovery near goal with width attacking advantage."
    }
]

for i, ex in enumerate(examples, 1):
    print(f"\n[Example {i}] {ex['club']}")
    print(f"  Scenario: {ex['scenario']}")
    print(f"  Zones: {ex['zones']}")
    print(f"  Shot Rate: {ex['shot_rate']*100:.0f}%")
    print(f"  Why: {ex['interpretation']}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

insights = [
    ("Zone Quality Matters", "Zones 10-11 (attacking territory) are 5-6× more dangerous than Zones 0-4 (defensive). Position > just winning the ball."),
    ("Central > Flanks", "Zone 10-11 center (0.19) beats Zone 9-10 flanks (0.14). Direct line to goal > wide play"),
    ("Transition Speed", "Zones 8-11 show 3-4× higher shot rates than mid-field because attacking momentum is preserved"),
    ("Flank Asymmetry", "Left flank (0.09-0.10) slightly lower than right in attacking 3rd - could indicate right-footed dominance"),
    ("Deep Defense Safety", "Zones 0-4 stay low (0.02-0.06) - deep clearances almost never lead to immediate shots"),
    ("Dead Zone", "Zone 4-6 midfield (0.03-0.04) is safest to defend - far from goal, time for reorganization"),
]

for title, insight in insights:
    print(f"\n✓ {title}")
    print(f"  → {insight}")

print("\n" + "=" * 80)
print("HOW TO USE THIS FOR DEFENSE")
print("=" * 80)

tactics = [
    ("If opponent recovers ball in Zone 10-11 (attacking center)", "PANIC MODE - 19% shot rate. Immediate defensive action needed. Clear heatmap shows this is most dangerous."),
    ("If opponent recovers ball in Zone 0-4 (your defense)", "Relatively SAFE - 3% shot rate. You have time to organize. Clearance is sufficient."),
    ("If opponent recovers ball in Zones 4-8 (midfield)", "MODERATE RISK - 3-5% shot rate. Encourage sideways/backwards play. Midfield is safest zone."),
    ("If pressuring opponent on the wing in hostile territory (Zone 9-10)", "HIGH REWARD for pressure win - 13-14% shot rate if you lose it. Make it count when you press!"),
]

for situation, recommendation in tactics:
    print(f"\n⚡ {situation}")
    print(f"   Action: {recommendation}")

print("\n" + "=" * 80)


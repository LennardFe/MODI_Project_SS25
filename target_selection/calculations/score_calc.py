def get_best_scoring_anchor(distance_changes, bearings, method):
    if method == "Ole":
        # Highest distance reduction and smallest bearing gets the best score
        best_distance_anchor = min(distance_changes, key=distance_changes.get)
        best_bearing_anchor = min(bearings, key=bearings.get)

        distange_changes_sorted = sorted(distance_changes.items(), key=lambda x: x[1])
        bearings_sorted = sorted(bearings.items(), key=lambda x: x[1])

        # Get second best distance change and bearing
        distance_difference_to_best = (
            distange_changes_sorted[1][1] - distange_changes_sorted[0][1]
        )
        distance_bearing_to_best = bearings_sorted[1][1] - bearings_sorted[0][1]

        # Calculate score
        # TODO Catch divison through zero
        distance_changes_gap = distance_difference_to_best / (
            distange_changes_sorted[-1][1] - distange_changes_sorted[0][1]
        )
        bearings_gap = distance_bearing_to_best / (
            bearings_sorted[-1][1] - bearings_sorted[0][1]
        )

        if distance_changes_gap > bearings_gap:
            return best_distance_anchor
        else:
            return best_bearing_anchor

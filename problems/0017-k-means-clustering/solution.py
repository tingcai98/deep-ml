import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	# return final_centroids
	points = np.array(points, dtype = float)
	initial_centroids = np.array(initial_centroids, dtype = float)
	for _ in range(max_iterations):
		clusters = [[] for _ in range(k)]
		# step 1: compute distance
		for p in points:
			distances = [np.linalg.norm(p - c) for c in initial_centroids]
			closest = np.argmin(distances)
			clusters[closest].append(p)

		# step 2: update initial_centroids
		new_centroids = initial_centroids.copy()

		for i in range(k):
			if clusters[i]:
				new_centroids[i] = np.mean(clusters[i], axis = 0)

		if np.array_equal(initial_centroids, new_centroids):
			break
		initial_centroids = new_centroids
	return [tuple(np.around(c, 4)) for c in new_centroids]
		
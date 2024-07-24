import matplotlib.pyplot as plt
hours_studied = [10, 9, 2, 15, 10, 16, 11, 16]
exam_scores = [95, 80, 10, 50, 45, 98, 38, 93]
plt.plot(hours_studied, exam_scores, marker='*', color='red', linestyle='-')
plt.xlabel('Number of hrs spent studying')
plt.ylabel('Score in the final exam (0 - 100)')
plt.title('Relationship between Hours Studied and Exam Scores')
plt.grid(True)
plt.show()
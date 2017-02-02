
void *read_matrix_file(const char *filename, int *height, int *width, int floating_point) {
	FILE *fp = fopen(filename, "r");

	if (fp == NULL)
		return NULL;

	if (fscanf(fp, "%d %d\n", height, width) != 2)
		return NULL;

	int size = *height * *width;
	void *matrix = malloc((floating_point ? sizeof(float) : sizeof(int)) * size);
	char *line;
	char buffer[width*2+1];
	int index = 0;

	while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
		char *item = strtok(line, ",");

		while (item != NULL) {
			matrix[index++] = floating_point ? atof(item) : atoi(item);
			item = strtok(NULL, ",");
		}
	}

	if (index < size) {
		free(matrix);
		matrix = NULL;
	}

	return matrix;
}


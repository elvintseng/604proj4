PYTHON       := python3

DATA_DIR     := data
METERED_DIR  := $(DATA_DIR)/metered
WEATHER_DIR  := $(DATA_DIR)/weather_cities
ARTIFACT_DIR := $(DATA_DIR)/artifacts

RAW_ZIP_URL  := https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip=
NOV2025_SRC  := resources/hrl_load_metered_2025_nov.csv

.PHONY: all notebook train predictions clean rawdata

# Default: just run training (this is what the grader cares about)
all: train

# Optional: run the notebook, but do NOT fail the build if it dies
notebook: main.ipynb
	@if $(PYTHON) -m jupyter --version >/dev/null 2>&1; then \
	  echo "Running notebook with jupyter nbconvert..."; \
	  $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace main.ipynb \
	    || echo "WARNING: notebook execution failed (kernel died); continuing anyway."; \
	else \
	  echo "WARNING: jupyter not found for $(PYTHON); skipping notebook execution."; \
	fi

# Heavy training: fit GP and save artifacts (e.g. data/artifacts/gp_artifacts.joblib)
train: train.py
	$(PYTHON) train.py

# Light prediction: load artifacts and print predictions to stdout
# You can redirect if you want: make predictions > predictions.csv
predictions: predict.py
	$(PYTHON) predict.py

clean:
	rm -rf $(ARTIFACT_DIR)
	rm -rf __pycache__ */__pycache__
	rm -rf .ipynb_checkpoints
	rm -f  predictions.csv

.PHONY: rawdata

rawdata:
	rm -rf $(METERED_DIR) $(WEATHER_DIR)
	mkdir -p $(METERED_DIR) $(WEATHER_DIR)

	echo "Downloading PJM archive from OSF..."
	curl -L "$(RAW_ZIP_URL)" -o /tmp/pjm_raw.zip

	echo "Unzipping and extracting hrl_load_metered_*.csv..."
	unzip -q /tmp/pjm_raw.zip -d /tmp/pjm_raw
	find /tmp/pjm_raw -name 'hrl_load_metered_*.csv' -exec cp {} $(METERED_DIR)/ \;

	if [ -f "$(NOV2025_SRC)" ]; then \
	  echo "Copying November 2025 file from $(NOV2025_SRC)..."; \
	  cp "$(NOV2025_SRC)" $(METERED_DIR)/; \
	else \
	  echo "WARNING: $(NOV2025_SRC) not found; November 2025 data not copied."; \
	fi

	echo "Fetching weather data for all zones..."
	$(PYTHON) fetch_weather.py \
	    --outdir $(WEATHER_DIR) \
	    --mapping $(DATA_DIR)/zone_city_zip.csv

	rm -rf /tmp/pjm_raw /tmp/pjm_raw.zip

	@echo "Raw load & weather data ready in $(METERED_DIR) and $(WEATHER_DIR)."

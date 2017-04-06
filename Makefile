.PHONY: help
help : Makefile
	@sed -n 's/^##//p' $<

pink121d = /home/kjbrown/labelreview/pink121
pink121_rhd_raw_csvs = $(wildcard $(pink121d)/*_edit.csv)
pink121_rhd_dats = $(patsubst %_edit.csv, %.dat, $(pink121_rhd_raw_csvs))
pink121_rhd_train_dats = $(patsubst $(pink121d)/%, train/pink121/%, $(pink121_rhd_dats))
## train_pink121	-	get pink121_rhd data for training
.PHONY: train_pink121
train_pink121: train/pink121/ test/pink121/ $(pink121_rhd_train_dats)
	# move test set
	mv ./train/pink121/2016-04-13-day-1436_raw-mic.* ./test/pink121/
test/pink121/:
	mkdir -p $@
train/pink121/:
	mkdir -p $@
train/pink121/%.csv.raw: $(pink121d)/%_edit.csv
	@#remove all z or blank labels
	cp $^.meta.yaml $@.meta.yaml
	python enrich_csv.py $^ $@
train/pink121/%.dat: $(pink121d)/%.dat train/pink121/%.csv.raw
	dat-enrich -w 0.4 $^ $@ 

black33d = /home/kjbrown/labelreview/black33
black33_rhd_raw_csvs = $(wildcard $(black33d)/*_edit.csv)
black33_rhd_dats = $(patsubst %_edit.csv, %.dat, $(black33_rhd_raw_csvs))
black33_rhd_train_dats = $(patsubst $(black33d)/%, train/black33/%, $(black33_rhd_dats))
## train_black33	-	get black33_rhd data for training
.PHONY: train_black33
train_black33: train/black33/ test/black33/ $(black33_rhd_train_dats)
	# move test set
	mv ./train/black33/2016-06-05-day-0906_raw-mic.* ./test/black33/
test/black33/:
	mkdir -p $@
train/black33/:
	mkdir -p $@
train/black33/%.csv.raw: $(black33d)/%_edit.csv
	@#remove all z or blank labels
	cp $^.meta.yaml $@.meta.yaml
	python enrich_csv.py $^ $@
train/black33/%.dat: $(black33d)/%.dat train/black33/%.csv.raw
	dat-enrich -w 0.4 $^ $@ 


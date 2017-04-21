.PHONY: help
help : Makefile
	@sed -n 's/^##//p' $<

.PHONY: all
## all - prep all raw data
all: train_black33 train_pink121 train_white17

sampledir=raw_training

pink121d = /home/kjbrown/labelreview/pink121
pink121_rhd_raw_csvs = $(wildcard $(pink121d)/*_edit.csv)
pink121_rhd_dats = $(patsubst %_edit.csv, %.dat, $(pink121_rhd_raw_csvs))
pink121_rhd_train_dats = $(patsubst $(pink121d)/%, $(sampledir)/pink121/%, $(pink121_rhd_dats))
## train_pink121	-	get pink121_rhd data for training
.PHONY: train_pink121
train_pink121: $(sampledir)/pink121/ $(pink121_rhd_train_dats)
$(sampledir)/pink121/:
	mkdir -p $@
$(sampledir)/pink121/%.csv.raw: $(pink121d)/%_edit.csv
	@#remove all z or blank labels
	cp $^.meta.yaml $@.meta.yaml
	python enrich_csv.py -b 0.03 -l y $^ $@
$(sampledir)/pink121/%.dat: $(pink121d)/%.dat $(sampledir)/pink121/%.csv.raw
	dat-enrich -w 20 $^ $@ 

black33d = /home/kjbrown/labelreview/black33
black33_rhd_raw_csvs = $(wildcard $(black33d)/*_edit.csv)
black33_rhd_dats = $(patsubst %_edit.csv, %.dat, $(black33_rhd_raw_csvs))
black33_rhd_train_dats = $(patsubst $(black33d)/%, $(sampledir)/black33/%, $(black33_rhd_dats))
## train_black33	-	get black33_rhd data for training
.PHONY: train_black33
train_black33: $(sampledir)/black33/ $(black33_rhd_train_dats)
$(sampledir)/black33/:
	mkdir -p $@
$(sampledir)/black33/%.csv.raw: $(black33d)/%_edit.csv
	@#remove all z or blank labels
	cp $^.meta.yaml $@.meta.yaml
	python enrich_csv.py -b 0.03 -l y $^ $@
$(sampledir)/black33/%.dat: $(black33d)/%.dat $(sampledir)/black33/%.csv.raw
	dat-enrich -w 20 $^ $@ 

white17d = /home/kjbrown/labelreview/white17
white17_rhd_raw_csvs = $(wildcard $(white17d)/*_edit.csv)
white17_rhd_dats = $(patsubst %_edit.csv, %.dat, $(white17_rhd_raw_csvs))
white17_rhd_train_dats = $(patsubst $(white17d)/%, $(sampledir)/white17/%, $(white17_rhd_dats))
## train_white17	-	get white17_rhd data for training
.PHONY: train_white17
train_white17: $(sampledir)/white17/ $(white17_rhd_train_dats)
$(sampledir)/white17/:
	mkdir -p $@
$(sampledir)/white17/%.csv.raw: $(white17d)/%_edit.csv
	@#remove all z or blank labels
	cp $^.meta.yaml $@.meta.yaml
	python enrich_csv.py -b 0.03 -l y $^ $@
$(sampledir)/white17/%.dat: $(white17d)/%.dat $(sampledir)/white17/%.csv.raw
	dat-enrich -w 20 $^ $@ 


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
	csvgrep -c name -r "(^z$$|^$$)" -i $^ > $@


train/pink121/%.dat: $(pink121d)/%.dat train/pink121/%.csv.raw
	dat-enrich -w 0.4 $^ $@ 


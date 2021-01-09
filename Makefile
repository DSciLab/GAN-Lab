PYTHON			:= python
PIP				:= pip
MAKE			:= make
CFG_DIR 		:= cfg
MLUTILS_DIR		:= mlutils
REQUIREMENTS	:= requirements.txt


.PHONY: all install dep


all: install


install_cfg: $(CFG_DIR)
	$(MAKE) -C $^ install


install_mlutils: $(MLUTILS_DIR)
	$(MAKE) -C $^ install


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


install: dep install_cfg install_mlutils

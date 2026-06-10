.PHONY: setup-redpanda build-cpp

setup-redpanda:
	docker exec -it redpanda rpk topic create raw-text-stream
	docker exec -it redpanda rpk topic create inference-telemetry

build-cpp:
	cd inference && python3 -m pip install "pybind11[global]" && python3 setup.py build_ext --inplace

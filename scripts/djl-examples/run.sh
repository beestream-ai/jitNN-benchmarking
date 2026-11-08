#!/bin/bash
# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# usage: run.sh <djl checkout> <jitnn|pytorch|onnxruntime> <example: mnist|lstm|resnet|classify|yolo|yolo-loop> [example args...]
# Copies DJL's example sources, builds, and runs one example on one engine. The training examples
# are copied unchanged. The two inference examples name their engine in the source ("PyTorch",
# "OnnxRuntime"); that one string becomes the default engine, nothing else changes.
set -e
djl=$1; engine=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
src=$here/src/main/java/ai/djl/examples/training
mkdir -p $src/util $src/transferlearning
cp $djl/examples/src/main/java/ai/djl/examples/training/{TrainMnist,TrainMnistWithLSTM}.java $src/
cp $djl/examples/src/main/java/ai/djl/examples/training/util/Arguments.java $src/util/
cp $djl/examples/src/main/java/ai/djl/examples/training/transferlearning/TrainResnetWithCifar10.java $src/transferlearning/
inf=$djl/examples/src/main/java/ai/djl/examples/inference/cv
mkdir -p $here/src/main/java/ai/djl/examples/inference/cv
for f in ImageClassification YoloDetection; do
    sed -e 's/Model.newInstance(modelName, "PyTorch")/Model.newInstance(modelName, ai.djl.engine.Engine.getDefaultEngineName())/' \
        -e 's/.optEngine("OnnxRuntime")/.optEngine(ai.djl.engine.Engine.getDefaultEngineName())/' \
        $inf/$f.java > $here/src/main/java/ai/djl/examples/inference/cv/$f.java
done
cd $here
# TrainResnetWithCifar10 classifies this image after training
mkdir -p src/test/resources && cp $djl/examples/src/test/resources/{airplane1.png,0.png,yolov8_test.jpg} src/test/resources/ 2>/dev/null || true
mvn=${MVN:-mvn}
$mvn -q -P$engine package dependency:build-classpath -Dmdep.outputFile=cp.txt -DskipTests > /dev/null
cp="target/classes:$(cat cp.txt)"
if [ "$engine" = jitnn ]; then
    exec java @$TORNADOVM_HOME/tornado-argfile ${JITNN_JVM_ARGS:-} -cp "$cp" bench.Driver "$@"
else
    exec java ${JVM_ARGS:-} -cp "$cp" bench.Driver "$@"
fi

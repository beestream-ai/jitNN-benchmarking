/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.bench;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.Test;

/** Profiles the workloads end to end on the test device; only with {@code -Djitnn.models=true} (see {@link ModelProfile}). */
class ModelProfileAccelTest {

    @Test
    void models() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.models"), "set -Djitnn.models=true");
        assertTrue(ModelProfile.run() > 0, "no workload ran");
    }
}

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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Samples the GPU's SM clock and throttle reasons every 100 ms through nvidia-smi, so each timed
 * window can be checked: accepted when no sample in it shows a power or thermal cap and the mean
 * SM clock is at least the threshold. Without nvidia-smi every window is accepted, unflagged.
 */
final class ClockMonitor implements AutoCloseable {

    /** Throttle reasons that slow the GPU: SW power cap, HW slowdown, SW and HW thermal, HW power brake. */
    static final long THROTTLE = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;

    record Sample(long nanos, int mhz, long reasons) {
    }

    record Window(int samples, double meanMhz, boolean throttled) {
    }

    private final List<Sample> samples = new ArrayList<>();
    private Process process;

    ClockMonitor(String gpu) {
        try {
            process = new ProcessBuilder("nvidia-smi", "-i", gpu, "--query-gpu=clocks.sm,clocks_event_reasons.active", "--format=csv,noheader,nounits", "-lms", "100")
                    .redirectErrorStream(true).start();
            Thread t = new Thread(() -> {
                try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        String[] f = line.split(",");
                        if (f.length < 2) {
                            continue;
                        }
                        try {
                            int mhz = Integer.parseInt(f[0].trim());
                            long reasons = Long.decode(f[1].trim());
                            synchronized (samples) {
                                samples.add(new Sample(System.nanoTime(), mhz, reasons));
                            }
                        } catch (NumberFormatException ignored) {
                            // header or message line
                        }
                    }
                } catch (Exception ignored) {
                    // the process ended
                }
            }, "clock-monitor");
            t.setDaemon(true);
            t.start();
        } catch (Exception e) {
            process = null;
        }
    }

    boolean available() {
        return process != null;
    }

    Window window(long from, long to) {
        int n = 0;
        double sum = 0;
        boolean throttled = false;
        synchronized (samples) {
            for (Sample s : samples) {
                if (s.nanos() >= from && s.nanos() <= to + 100_000_000L) {
                    n++;
                    sum += s.mhz();
                    throttled |= (s.reasons() & THROTTLE) != 0;
                }
            }
            samples.removeIf(s -> s.nanos() < from - 2_000_000_000L);
        }
        return new Window(n, n == 0 ? 0 : sum / n, throttled);
    }

    @Override
    public void close() {
        if (process != null) {
            process.destroy();
        }
    }
}

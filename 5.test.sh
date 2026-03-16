llama.cpp/build/bin/llama-cli \
          -m llama2-merged-q2_k_m.gguf \
            -ngl 0 \
              -c 1024 \
                -n 128 \
                  -p "Explain attention briefly."

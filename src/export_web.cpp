#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: nexus_export_web <checkpoint.pt> <tokenizer.bpe> <output-directory>\n";
            return 2;
        }
        nexus::BpeTokenizer tokenizer;
        tokenizer.load(argv[2]);
        nexus::ModelConfig cfg;
        cfg.vocab_size = tokenizer.vocab_size();
        cfg.context = 256;
        nexus::NexusModel model(cfg);
        torch::load(model, argv[1]);
        model->to(torch::kCPU);
        model->eval();

        const std::filesystem::path out_dir(argv[3]);
        std::filesystem::create_directories(out_dir);
        std::ofstream weights(out_dir / "model.bin", std::ios::binary | std::ios::trunc);
        std::ofstream manifest(out_dir / "model.json", std::ios::trunc);
        if (!weights || !manifest) throw std::runtime_error("cannot create web model output files");

        manifest << "{\n  \"format\": \"NEXUS_WEB_V1\",\n"
                 << "  \"config\": {\"vocab_size\": " << cfg.vocab_size
                 << ", \"width\": " << cfg.width << ", \"layers\": " << cfg.layers
                 << ", \"heads\": " << cfg.heads << ", \"context\": " << cfg.context
                 << ", \"experts\": " << cfg.experts << ", \"top_k\": " << cfg.top_k << "},\n"
                 << "  \"tensors\": [\n";
        auto parameters = model->named_parameters(true);
        std::uint64_t offset = 0;
        std::size_t index = 0;
        bool has_lm_head = false;
        std::uint64_t embedding_offset = 0;
        std::string embedding_shape;
        for (const auto& item : parameters) {
            auto tensor = item.value().detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            const auto bytes = static_cast<std::uint64_t>(tensor.numel()) * sizeof(float);
            weights.write(reinterpret_cast<const char*>(tensor.data_ptr<float>()),
                          static_cast<std::streamsize>(bytes));
            if (!weights) throw std::runtime_error("failed writing model weights");
            if (item.key() == "lm_head.weight") has_lm_head = true;
            if (item.key() == "embedding.weight") {
                embedding_offset = offset;
                for (int64_t d = 0; d < tensor.dim(); ++d) {
                    if (d) embedding_shape += ", ";
                    embedding_shape += std::to_string(tensor.size(d));
                }
            }
            manifest << "    {\"name\": \"" << item.key() << "\", \"shape\": [";
            for (int64_t d = 0; d < tensor.dim(); ++d) {
                if (d) manifest << ", ";
                manifest << tensor.size(d);
            }
            manifest << "], \"offset\": " << offset << ", \"length\": " << bytes << "}";
            if (++index != parameters.size() || !has_lm_head) manifest << ",";
            manifest << "\n";
            offset += bytes;
        }
        if (!has_lm_head) {
            if (embedding_shape.empty()) throw std::runtime_error("tied embedding tensor is missing");
            manifest << "    {\"name\": \"lm_head.weight\", \"shape\": ["
                     << embedding_shape << "], \"offset\": " << embedding_offset
                     << ", \"length\": " << (cfg.vocab_size * cfg.width * sizeof(float)) << "}\n";
        }
        manifest << "  ]\n}\n";
        weights.flush();
        if (!manifest || !weights) throw std::runtime_error("failed writing browser model files");
        std::filesystem::copy_file(argv[2], out_dir / "tokenizer.bpe",
                                   std::filesystem::copy_options::overwrite_existing);
        std::cout << "Exported " << index << " tensors (" << offset
                  << " bytes) to " << out_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NEXUS web export error: " << e.what() << "\n";
        return 1;
    }
}

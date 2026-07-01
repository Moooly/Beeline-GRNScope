# Include packages

using NetworkInference
using LightGraphs
using SparseArrays

algorithm = PIDCNetworkInference()

dataset_name = string(ARGS[1])
out_file = string(ARGS[2])
matrix_format = length(ARGS) >= 3 ? lowercase(string(ARGS[3])) : "auto"

function detect_delimiter(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        return occursin(",", header) ? ',' : '\t'
    end
end

function parse_sparse_node(parts, discretizer, estimator, number_of_bins)
    label = parts[1]
    nonzero_indices = Int[]
    nonzero_values = Float64[]

    for idx in 2:length(parts)
        value = parse(Float64, parts[idx])
        if value != 0.0
            push!(nonzero_indices, idx - 1)
            push!(nonzero_values, value)
        end
    end

    sparse_values = sparsevec(nonzero_indices, nonzero_values, length(parts) - 1)
    node_line = Vector{Any}(undef, length(parts))
    node_line[1] = label
    node_line[2:end] = Vector{Float64}(sparse_values)
    return Node(node_line, discretizer, estimator, number_of_bins)
end

function parse_dense_node(parts, discretizer, estimator, number_of_bins)
    node_line = Vector{Any}(undef, length(parts))
    node_line[1] = parts[1]
    for idx in 2:length(parts)
        node_line[idx] = parse(Float64, parts[idx])
    end
    return Node(node_line, discretizer, estimator, number_of_bins)
end

function get_nodes_streaming(
    path::AbstractString;
    delim = detect_delimiter(path),
    matrix_format = "auto",
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
)
    nodes = Node[]
    parse_sparse = matrix_format in ("auto", "sparse", "true", "yes", "on", "1")

    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(chomp(line), delim)
            node = parse_sparse ?
                parse_sparse_node(parts, discretizer, estimator, number_of_bins) :
                parse_dense_node(parts, discretizer, estimator, number_of_bins)
            push!(nodes, node)
        end
    end
    return nodes
end

@time genes = get_nodes_streaming(dataset_name, matrix_format = matrix_format);

@time network = InferredNetwork(algorithm, genes);

write_network_file(out_file, network);

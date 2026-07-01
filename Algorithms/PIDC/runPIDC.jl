# Include packages

using NetworkInference
using LightGraphs
algorithm = PIDCNetworkInference()

dataset_name = string(ARGS[1])
out_file = string(ARGS[2])
requested_matrix_format = length(ARGS) >= 3 ? lowercase(string(ARGS[3])) : "auto"
if requested_matrix_format in ("sparse", "true", "yes", "on", "1")
    @warn "PIDC streams the input file, but NetworkInference requires dense per-gene Node vectors."
end

function detect_delimiter(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        return occursin(",", header) ? ',' : '\t'
    end
end

function parse_values(parts, line_number)
    if length(parts) <= 1
        throw(ArgumentError("Expression row $line_number has no cell values. Check the input delimiter."))
    end

    values = Vector{Float64}(undef, length(parts) - 1)
    for idx in 2:length(parts)
        value_text = strip(parts[idx])
        if isempty(value_text)
            throw(ArgumentError("Expression row $line_number has an empty value at column $idx."))
        end
        values[idx - 1] = parse(Float64, value_text)
    end

    if any(value -> !isfinite(value), values)
        throw(ArgumentError("Expression row $line_number contains NaN or Inf values."))
    end
    return values
end

function is_informative(values)
    isempty(values) && return false
    first_value = values[1]
    return any(value -> value != first_value, values)
end

function parse_node(parts, line_number, discretizer, estimator, number_of_bins)
    label = strip(parts[1])
    isempty(label) && throw(ArgumentError("Expression row $line_number has an empty gene name."))

    values = parse_values(parts, line_number)
    if !is_informative(values)
        return nothing
    end

    node_line = Matrix{Any}(undef, 1, length(parts))
    node_line[1, 1] = label
    for idx in 1:length(values)
        node_line[1, idx + 1] = values[idx]
    end
    return Node(node_line, discretizer, estimator, number_of_bins)
end

function get_nodes_streaming(
    path::AbstractString;
    delim = detect_delimiter(path),
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
)
    nodes = Node[]
    skipped_constant = 0

    open(path, "r") do io
        header = split(chomp(readline(io)), delim, keepempty = true)
        expected_width = length(header)
        if expected_width <= 1
            throw(ArgumentError("Expression header has no cell columns. Check the input delimiter."))
        end

        for (offset, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            line_number = offset + 1
            parts = split(chomp(line), delim, keepempty = true)
            if length(parts) != expected_width
                throw(ArgumentError(
                    "Expression row $line_number has $(length(parts)) columns; expected $expected_width."
                ))
            end

            node = parse_node(parts, line_number, discretizer, estimator, number_of_bins)
            if node === nothing
                skipped_constant += 1
            else
                push!(nodes, node)
            end
        end
    end

    if skipped_constant > 0
        @warn "PIDC skipped $skipped_constant genes with constant expression across all cells."
    end
    if isempty(nodes)
        throw(ArgumentError("PIDC found no informative genes after filtering constant rows."))
    end
    return nodes
end

@time genes = get_nodes_streaming(dataset_name);

@time network = InferredNetwork(algorithm, genes);

write_network_file(out_file, network);

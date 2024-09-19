"""
isinaltinkaya
Modified from original script: github.com/ZhiGroup/IBD_benchmark/Simulation/msprime_simulation.py
Changes:
    - separated main function into functions with specific purposes
    - added flexible population sampling
    - added flexible ancestry and mutation model sampling
    - removed unused variables
    - improved logging
    - added code comments explaining existing code functionality
    - improved speed by reducing file I/O operations

Original text:
Script to simulate dataset and ground truth identity by descents (IBDs) using msprime (version 1.x).
The demographic model used is the Gutenkunst et al. Out-of-Africa model.
"""
import msprime
import tskit
import numpy
import sys
import math
from bisect import bisect_right, bisect_left


"""Get physical and genetic positions from the genetic map file in HapMap format
-> Read the genetic map in HapMap format from file_input_name 
and extract physical (Position(bp) column) and genetic positions (Map(cM) column)

e.g. input file:
Chromosome  Position(bp)    Rate(cM/Mb) Map(cM) 
chr20   61795   0.735716    0.000000
chr20   63231   0.734243    0.001056
chr20   63244   0.734147    0.001066
chr20   63799   0.734513    0.001473

Returns:
    list: physical_positions
    list: genetic_positions
"""
def read_genetic_map_hapmap(file_input_name):
    physical_positions = []
    genetic_positions = []
    with open(file_input_name, "r") as file_input:
        for line_number, file_input_line in enumerate(file_input):
            if line_number == 0:
                pass
            else:
                file_input_line_tokens = file_input_line.strip().split("\t")
            if line_number > 0 and len(file_input_line_tokens) >= 4:
                physical_positions.append(int(file_input_line_tokens[1]))
                genetic_positions.append(float(file_input_line_tokens[3]))
    return physical_positions, genetic_positions


"""Read VCF file and create genetic position map with interpolated genetic positions
-> Read the physical positions from the simulated vcf (file_input_name)
-> Create genetic position map by linearly interpolating the genetic positions based on the physical positions extracted from the VCF
-> Why: the genetic map does not include all the physical positions in the VCF, so we need to interpolate the genetic positions for the missing physical positions to create a complete genetic position map
-> Write the map to file (map_output_name)

Returns:
    dict: genetic position map
"""
def read_vcf_and_get_genetic_map(file_input_name, map_output_name, physical_positions, genetic_positions):
    genetic_position_map = {}
    vcf_physical_positions = []
    with open(file_input_name, "r") as file_input:
        for line_number, file_input_line in enumerate(file_input):
            if file_input_line.startswith("#"):
                pass
            else:
                file_input_line_tokens = file_input_line.strip().split("\t")
                if len(file_input_line_tokens) >= 9:
                    vcf_physical_positions.append(int(file_input_line_tokens[1]))
    vcf_genetic_positions = numpy.interp(vcf_physical_positions, physical_positions, genetic_positions)
    for physical_position, genetic_position in zip(vcf_physical_positions, vcf_genetic_positions):
        genetic_position_map[physical_position] = genetic_position
    with open(map_output_name, "w") as map_output:
        for physical_position, genetic_position in zip(vcf_physical_positions, vcf_genetic_positions):
            map_output.write(f"{physical_position}\t{genetic_position}\n")
    return genetic_position_map



"""Find the true IBD segments among the tree sequences
-> The contiguous segments among the tree sequences where the haplotype pairs share the same MRCA are identified as the “true IBD segments”. The true IBD segments are selected if their genetic lengths were at least 1 cM (minimum_genetic_length) among the segments detected using the trees sampled for every 5000 base pair physical distance (physical_distance_to_sample).
-> [tree_info_last] Dictionary to keep track of the latest MRCA and the starting position of the segment for each pair of haplotypes
-> Iterate over the trees in the tree sequence
-> For each pair of haplotypes, get MRCA using tree.mrca(haplotype_id_2, haplotype_id_1). If the MRCA is different from the previous MRCA, then the haplotypes no longer share an IBD segment (end of the segment). The genetic length of the segment is calculated using the genetic_position_map with the physical positions of the start and end of the segment. If the genetic length is greater than or equal to the minimum_genetic_length, the segment is written to the output file.

Returns:
    None
"""
def find_true_ibds(file_output_name, chromosome_id, haplotype_ids, tree_sequence, minimum_genetic_length, physical_distance_to_sample, genetic_position_map):
    with open(file_output_name, "w") as ibd_file:
        ibd_file.write(f"#individual_1_id,individual_1_haplotype_id,individual_2_id,individual_2_haplotype_id,chromosome_id,true_ibd_physical_position_start,true_ibd_physical_position_end,genetic_length\n")
        tree_info_last = {}
        number_of_trees = tree_sequence.num_trees
        last_sampled_tree_genomic_region_end_physical_location = 0.0
        first_tree_site_current_tree = -1
        last_tree_site_previous_tree = -1
        for tree_id, tree in enumerate(tree_sequence.trees()):
            tree_sites = [round(tree_site.position) if tree_site is not None else -1 for tree_site in tree.sites()]
            first_tree_site_current_tree = tree_sites[0] if len(tree_sites) > 0 else first_tree_site_current_tree
            last_tree_site_current_tree = tree_sites[len(tree_sites) - 1] if len(tree_sites) > 0 else last_tree_site_previous_tree
            if tree_id == 0:
                for i in numpy.arange(0, len(haplotype_ids), 1):
                    haplotype_id_1 = haplotype_ids[i]
                    for j in numpy.arange(i + 1, len(haplotype_ids), 1):
                        haplotype_id_2 = haplotype_ids[j]
                        tree_info_last[(haplotype_id_2, haplotype_id_1)] = (tree.mrca(haplotype_id_2, haplotype_id_1), first_tree_site_current_tree)
                last_sampled_tree_genomic_region_end_physical_location = tree.interval[1]
            else:
                if tree.interval[1] - tree.interval[0] >= physical_distance_to_sample or tree.interval[1] - last_sampled_tree_genomic_region_end_physical_location >= physical_distance_to_sample or tree_id == number_of_trees - 1:
                    for i in numpy.arange(0, len(haplotype_ids), 1):
                        haplotype_id_1 = haplotype_ids[i]
                        for j in numpy.arange(i + 1, len(haplotype_ids), 1):
                            haplotype_id_2 = haplotype_ids[j]
                            mrca = tree.mrca(haplotype_id_2, haplotype_id_1)
                            if mrca != tree_info_last[(haplotype_id_2, haplotype_id_1)][0] or tree_id == number_of_trees - 1:
                                site_physical_position_start = tree_info_last[(haplotype_id_2, haplotype_id_1)][1]
                                site_physical_position_end = last_tree_site_previous_tree if tree_id < number_of_trees - 1 else last_tree_site_current_tree
                                if site_physical_position_start != -1 and site_physical_position_end != -1:
                                    ibd_genetic_length = genetic_position_map[site_physical_position_end] - genetic_position_map[site_physical_position_start]
                                    if ibd_genetic_length >= minimum_genetic_length:
                                        individual_1_id = haplotype_id_1 // 2
                                        individual_1_haplotype_id = haplotype_id_1 % 2
                                        individual_2_id = haplotype_id_2 // 2
                                        individual_2_haplotype_id = haplotype_id_2 % 2
                                        ibd_file.write(f"{individual_2_id},{individual_2_haplotype_id},{individual_1_id},{individual_1_haplotype_id},{chromosome_id},{site_physical_position_start},{site_physical_position_end},{ibd_genetic_length}\n")
                                tree_info_last[(haplotype_id_2, haplotype_id_1)] = (mrca, first_tree_site_current_tree)
                    last_sampled_tree_genomic_region_end_physical_location = tree.interval[1]
            last_tree_site_previous_tree = last_tree_site_current_tree
    return None


""" Simulate trees using msprime with the given parameters

Returns:
    None
"""
def simulate_trees(chromosome_id, 
        input_map_file,
        mutation_rate,
        random_seed,
        individuals_from_populations_to_sample_dictionary,
        sim_ancestry_model,
        sim_mutation_model,
        output_tree_file,
        log_file
        ):

    with open(str(log_file),'w') as log:
        sys.stdout=log

        print(f"start simulation")

        print(f"chromosome_id={chromosome_id},input_map_file={input_map_file},mutation_rate={mutation_rate},random_seed={random_seed},sim_ancestry_model={sim_ancestry_model},sim_mutation_model={sim_mutation_model},output_tree_file={output_tree_file}, log_file={log_file}")

        mutation_rate_value = float(mutation_rate)
        random_seed_value = int(random_seed)

        print(f"start load recombination map")
        recombination_rate_map = msprime.RateMap.read_hapmap(input_map_file, has_header=True, position_col=1, rate_col=2, map_col=None)
        print(f"end load recombination map")

        print(f"start build population model")
        # Gutenkunst et al. Out-of-Africa model
        out_of_africa_model_demography = msprime.Demography()
        generation_time = 25
        T_OOA = 21.2e3 / generation_time
        T_AMH = 140e3 / generation_time
        T_ANC = 220e3 / generation_time
        r_CEU = 0.004
        r_CHB = 0.0055
        N_CEU = 1000 / math.exp(-r_CEU * T_OOA)
        N_CHB = 510 / math.exp(-r_CHB * T_OOA)
        N_AFR = 12300
        out_of_africa_model_demography.add_population(name="YRI", description="Yoruba in Ibadan, Nigeria", initial_size=N_AFR)
        out_of_africa_model_demography.add_population(name="CEU", description="Utah Residents (CEPH) with Northern and Western European Ancestry", initial_size=N_CEU, growth_rate=r_CEU)
        out_of_africa_model_demography.add_population(name="CHB", description="Han Chinese in Beijing, China", initial_size=N_CHB, growth_rate=r_CHB)
        out_of_africa_model_demography.add_population(name="OOA", description="Bottleneck out-of-Africa population", initial_size=2100)
        out_of_africa_model_demography.add_population(name="AMH", description="Anatomically modern humans", initial_size=N_AFR)
        out_of_africa_model_demography.add_population(name="ANC", description="Ancestral equilibrium population", initial_size=7300)
        out_of_africa_model_demography.set_symmetric_migration_rate(["CEU", "CHB"], 9.6e-5)
        out_of_africa_model_demography.set_symmetric_migration_rate(["YRI", "CHB"], 1.9e-5)
        out_of_africa_model_demography.set_symmetric_migration_rate(["YRI", "CEU"], 3e-5)
        out_of_africa_model_demography.add_population_split(time=T_OOA, derived=["CEU", "CHB"], ancestral="OOA")
        out_of_africa_model_demography.add_symmetric_migration_rate_change(time=T_OOA, populations=["YRI", "OOA"], rate=25e-5)
        out_of_africa_model_demography.add_population_split(time=T_AMH, derived=["YRI", "OOA"], ancestral="AMH")
        out_of_africa_model_demography.add_population_split(time=T_ANC, derived=["AMH"], ancestral="ANC")
        print(f"end build population model")

        print(f"start simulate data")

        tree_sequence_without_mutations = msprime.sim_ancestry(samples=individuals_from_populations_to_sample_dictionary, demography=out_of_africa_model_demography, ploidy=2, discrete_genome=True, recombination_rate=recombination_rate_map, random_seed=random_seed_value, record_provenance=False, model=sim_ancestry_model)
        tree_sequence = msprime.sim_mutations(tree_sequence_without_mutations, rate=mutation_rate_value, random_seed=random_seed_value, model=sim_mutation_model, discrete_genome=True)
        tree_sequence.dump(output_tree_file)

        print(f"number_of_trees={tree_sequence.num_trees}")

        print(f"end simulate data")

    return None


"""Convert tree sequence to VCF format

Returns:
    None
"""
def tree_to_vcf(input_tree_sequence_file,chromosome_id,output_vcf_file,log_file):

    with open(str(log_file),'w') as log:
        sys.stdout=log

        print(f"load tree from file {input_tree_sequence_file}")
        tree_sequence=tskit.load(input_tree_sequence_file)

        print(f"start generate raw vcf file")
        individual_ids = [str(individual_id) for individual_id in numpy.arange(int(tree_sequence.num_samples / 2))]
        with open(output_vcf_file, "w") as vcf_file:
            tree_sequence.write_vcf(vcf_file, individual_names=individual_ids, contig_id=chromosome_id, position_transform="legacy")

        print(f"number_of_sites_raw={tree_sequence.num_sites}")
        print(f"end generate raw vcf file")
    
    return None


"""Find the ground truth IBD segments among the tree sequences
-> Wrapper function to find the ground truth IBD segments from the tree sequences

Returns:
    None
"""
def find_ground_truth_ibds(chromosome_id,physical_distance_to_sample,input_tree_sequence_file,input_vcf_file,input_map_file, output_raw_vcf_true_ibd_file , output_raw_map_file, minimum_genetic_length,log_file):

    with open(str(log_file),'w') as log:
        sys.stdout=log

        print(f"load tree from file {input_tree_sequence_file}")
        tree_sequence=tskit.load(input_tree_sequence_file)

        minimum_genetic_length_value = float(minimum_genetic_length)
        print(f"minimum_genetic_length={minimum_genetic_length}")

        physical_positions, genetic_positions = read_genetic_map_hapmap(input_map_file)

        physical_distance_to_sample_value = int(physical_distance_to_sample)

        print(f"start find ground truth ibds of raw vcf file")
        haplotype_ids = numpy.arange(tree_sequence.num_samples)
        vcf_raw_genetic_position_map = read_vcf_and_get_genetic_map(input_vcf_file, output_raw_map_file, physical_positions, genetic_positions)

        find_true_ibds(output_raw_vcf_true_ibd_file, chromosome_id, haplotype_ids, tree_sequence, minimum_genetic_length_value, physical_distance_to_sample_value, vcf_raw_genetic_position_map)

        print(f"end find ground truth ibds of raw vcf file")
        print(f"end find_ground_truth_ibds")
        print(f"end simulation")

    return None


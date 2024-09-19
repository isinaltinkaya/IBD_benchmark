"""
Script to simulate dataset and ground truth identity by descents (IBDs) using msprime (version 1.x).
The demographic model used is the Gutenkunst et al. Out-of-Africa model.
The individuals are sampled from European population.
"""
import msprime
import numpy
import sys
import math
from bisect import bisect_right, bisect_left


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
                                    with open(file_output_name, "a") as ibd_file:
                                        ibd_file.write(f"{individual_2_id},{individual_2_haplotype_id},{individual_1_id},{individual_1_haplotype_id},{chromosome_id},{site_physical_position_start},{site_physical_position_end},{ibd_genetic_length}\n")
                            tree_info_last[(haplotype_id_2, haplotype_id_1)] = (mrca, first_tree_site_current_tree)
                last_sampled_tree_genomic_region_end_physical_location = tree.interval[1]
        last_tree_site_previous_tree = last_tree_site_current_tree
    return None



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


if __name__ == '__main__':
    print(f"start simulation")

    print(f"start get arguments")
    program_name = sys.argv[0]
    program_arguments = sys.argv[1:]
    chromosome_id = program_arguments[0]
    input_map_file = program_arguments[1]
    mutation_rate = program_arguments[2]
    random_seed = program_arguments[3]
    number_of_individuals_to_sample = program_arguments[4]
    number_of_sites_array_template = program_arguments[5]
    physical_distance_to_sample = program_arguments[6]
    minimum_genetic_length = program_arguments[7]
    output_directory_path = program_arguments[8]

    print(f"program={program_name},chromosome_id={chromosome_id},input_map_file={input_map_file},mutation_rate={mutation_rate},random_seed={random_seed},number_of_individuals_to_sample={number_of_individuals_to_sample},number_of_sites_array_template={number_of_sites_array_template},physical_distance_to_sample={physical_distance_to_sample},minimum_genetic_length={minimum_genetic_length},output_directory_path={output_directory_path}")

    mutation_rate_value = float(mutation_rate)
    random_seed_value = int(random_seed)
    number_of_individuals_to_sample_value = int(number_of_individuals_to_sample)
    number_of_sites_array_template_value = int(number_of_sites_array_template)
    physical_distance_to_sample_value = int(physical_distance_to_sample)
    minimum_genetic_length_value = float(minimum_genetic_length)
    output_tree_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.trees"
    output_raw_vcf_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.vcf"
    output_raw_map_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.fit.map"
    output_biallelic_vcf_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.biallelic.vcf"
    output_biallelic_map_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.biallelic.fit.map"
    output_array_vcf_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.array.vcf"
    output_array_map_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.array.fit.map"
    output_raw_vcf_true_ibd_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.ti{minimum_genetic_length}.txt"
    output_biallelic_vcf_true_ibd_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.biallelic.ti{minimum_genetic_length}.txt"
    output_array_vcf_true_ibd_file = output_directory_path + f"ooa{number_of_individuals_to_sample}.chr{chromosome_id}.array.ti{minimum_genetic_length}.txt"
    print(f"end get arguments")

    print(f"start load recombination map")
    physical_positions, genetic_positions = read_genetic_map_hapmap(input_map_file)
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
    individuals_from_populations_to_sample_dictionary = {"CEU": number_of_individuals_to_sample_value}
    tree_sequence_without_mutations = msprime.sim_ancestry(samples=individuals_from_populations_to_sample_dictionary, demography=out_of_africa_model_demography, ploidy=2, discrete_genome=True, recombination_rate=recombination_rate_map, random_seed=random_seed_value, record_provenance=False, model="hudson")
    tree_sequence = msprime.sim_mutations(tree_sequence_without_mutations, rate=mutation_rate_value, random_seed=random_seed_value, model=msprime.JC69(), discrete_genome=True)
    tree_sequence.dump(output_tree_file)
    print(f"number_of_trees={tree_sequence.num_trees}")
    print(f"end simulate data")

    print(f"start generate raw vcf file")
    individual_ids = [str(individual_id) for individual_id in numpy.arange(int(tree_sequence.num_samples / 2))]
    with open(output_raw_vcf_file, "w") as vcf_file:
        tree_sequence.write_vcf(vcf_file, individual_names=individual_ids, contig_id=chromosome_id, position_transform="legacy")
    print(f"number_of_sites_raw={tree_sequence.num_sites}")
    print(f"end generate raw vcf file")


    print(f"start find ground truth ibds of raw vcf file")
    haplotype_ids = numpy.arange(tree_sequence.num_samples)
    vcf_raw_genetic_position_map = read_vcf_and_get_genetic_map(output_raw_vcf_file, output_raw_map_file, physical_positions, genetic_positions)
    find_true_ibds(output_raw_vcf_true_ibd_file, chromosome_id, haplotype_ids, tree_sequence, minimum_genetic_length_value, physical_distance_to_sample_value, vcf_raw_genetic_position_map)
    print(f"end find ground truth ibds of raw vcf file")


    print(f"end simulation")
    sys.exit(0)

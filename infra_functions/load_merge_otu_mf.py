import pandas as pd


class OtuMfHandler:
    def __init__(self, otu_csv_file_path, mapping_csv_file_path, from_QIIME=False, id_col='#OTU ID',
                 taxonomy_col='taxonomy', tax_in_col=False, make_merge=True, drop_rare_bacteria=False):
        self.id_col = id_col
        self.taxonomy_col = taxonomy_col
        self.from_QIIME = from_QIIME
        self.mapping_file_path = mapping_csv_file_path
        self.otu_file_path = otu_csv_file_path
        self.tax_in_col = tax_in_col
        self.make_merge = make_merge
        self.drop_rare_bacteria = drop_rare_bacteria
        self.mapping_file, self.otu_file = self._load_data()
        self.otu_file_wo_taxonomy = self.get_otu_file_wo_taxonomy()
        if self.make_merge:
            self.merged_data = self._merge_otu_mf()

    def _load_data(self):
        mapping_file = pd.read_csv(self.mapping_file_path)
        mapping_file = mapping_file.set_index('#SampleID').sort_index()
        skip_rows = 0
        if self.from_QIIME:
            skip_rows = 1
        otu_file = pd.read_csv(self.otu_file_path, skiprows=skip_rows).set_index(self.id_col).T
        if self.tax_in_col:
            otu_file = otu_file.T
        if self.drop_rare_bacteria:
            """
            bact_name_to_id_dict = {}
            with open("bact_id_to_name_dict.txt", "r") as f:
                lines = f.readlines()
            for line in lines:
                id_bact = line.split(",")
                bact_name_to_id_dict[id_bact[1]] = id_bact[0]

            rare_ids = []
            for rare_bact in self.drop_rare_bacteria:
                if rare_bact in bact_name_to_id_dict.keys():
                    rare_ids.append(bact_name_to_id_dict[rare_bact])
            """
            otu_file.drop(self.drop_rare_bacteria, axis=1, inplace=True)


        return mapping_file, otu_file

    def _merge_otu_mf(self):
        merged_data = self.otu_file.join(self.mapping_file).T
        return merged_data

    def get_otu_file_wo_taxonomy(self):
        """
        :return: otu file without the taxonomy
        """

        tmp_copy = self.otu_file.T.copy()
        return tmp_copy.drop([self.taxonomy_col], axis=1).T

    def merge_mf_with_new_otu_data(self, new_otu_data):
        """
        :param new_otu_data:  new otu data columns are the bacterias and rows are the samples
        :return: new otu data with the original mapping file
        """
        tmp_copy = new_otu_data.copy()
        merged_data = tmp_copy.join(self.mapping_file)
        return merged_data

    def add_taxonomy_col_to_new_otu_data(self, new_otu_data):
        """
        :param new_otu_data: new otu data columns are the bacterias and rows are the samples
        :return: returns the new otu_data with the taxonomy col from the original otu file
        """
        tmp_copy = new_otu_data.T.copy().astype(object)
        tmp_copy[self.taxonomy_col] = self.otu_file[self.taxonomy_col]
        return tmp_copy

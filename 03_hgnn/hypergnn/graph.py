import torch

from collections import defaultdict
from torch import Tensor
import re

class Graph:

    RDF_TYPE = 'label'

    def __init__(self) -> None:
        self.graph_triples: list = None
        self.label_triples: list = None #new
        self.node2types_dict: dict = defaultdict(set)
        self.enum_nodes: dict = None
        self.enum_relations: dict = None
        self.enum_classes: list = None
        self.edge_index: Tensor = None
        self.edge_type: Tensor = None
        self.edge_index_label: Tensor = None
        self.edge_type_label: Tensor = None


    def get_graph_triples(self, file_path: str) -> None:
        with open(file_path, 'r') as file:
            self.graph_triples = file.read().splitlines()

    def get_label_triples(self,label_path: str) -> None:
        #new01
        with open(label_path, 'r') as file:
            self.label_triples = file.read().splitlines()
            

    def create_edge_data(self):
        '''create edge_index and edge_type'''
        
        edge_list: list = []
        for triple in self.graph_triples:
            # triple_list = triple.split('\t', maxsplit=2)
            triple_list = re.split(r'\s+', triple, maxsplit=2)
            if triple_list != ['']:
                s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()

                if self.enum_relations.get(p) != None:

                    # create edge list and also add inverse edge of each edge
                    src, dst, rel = self.enum_nodes[s], self.enum_nodes[o], self.enum_relations[p]
                    edge_list.append([src, dst, 2 * rel])
                    edge_list.append([dst, src, 2 * rel + 1])
   
        edges = torch.tensor(edge_list, dtype=torch.long).t() # shape(3, (2*number_of_edges - #RDF_TYPE_edges))
        self.edge_index = edges[:2] 
        self.edge_type = edges[2]

    # new02
    # def create_edge_data_label(self):
    #     '''create edge_index and edge_type'''
    #
    #     edge_list: list = []
    #     for triple in self.label_triples:
    #         triple_list = triple.split('\t', maxsplit=2)
    #         if triple_list != ['']:
    #             s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
    #             if self.enum_relations.get(p) != None:
    #                 # create edge list and also add inverse edge of each edge
    #                 src, dst, rel = self.enum_nodes[s], self.enum_nodes[o], self.enum_relations[p]
    #                 edge_list.append([src, dst, 2 * rel])
    #                 edge_list.append([dst, src, 2 * rel + 1])
    #
    #     edges = torch.tensor(edge_list, dtype=torch.long).t()  # shape(3, (2*number_of_edges - #RDF_TYPE_edges))
    #     self.edge_index_label = edges[:2]
    #     self.edge_type_label = edges[2]


    def init_graph(self, file_path: str) -> None:
        '''intialize graph object by creating and storing important graph variables'''

        self.get_graph_triples(file_path)
        
        # to store all subjects, predicates and objects 
        subjects = set()
        predicates = set()
        objects = set()

        class_count = defaultdict(int)

        # loop over each triple and split 2 times on space:' '
        for triple in self.graph_triples:
            triple_list = re.split(r'\s+', triple, maxsplit=2)
            # triple_list = triple.split('\t', maxsplit=2)
            # skip triple if there is a blank lines in .nt files
            if triple_list != ['']:
                s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
                # add nodes and predicates
                subjects.add(s)
                predicates.add(p)
                objects.add(o)
                 # check if subject is a valid entity and check if predicate is rdf:type
                # if str(p) == self.RDF_TYPE.lower():
                #         class_count[str(o)] += 1
                #         self.node2types_dict[s].add(o)

        # create a list with all nodes and enumerate the nodes
        nodes = list(subjects.union(objects))
        self.enum_nodes = {node: i for i, node in enumerate(sorted(nodes))}
        # remove the rdf:type relations since we would like to predict the types
        # and enumerate the relations and save as dict
        #predicates.remove(self.RDF_TYPE.lower())
        self.enum_relations = {rel: i for i, rel in enumerate(sorted(predicates))}
        # enumereate classes
        # self.enum_classes = {lab: i for i, lab in enumerate(class_count.keys())}

        # print class occurence dict too get insight in class (im)balance
        # print(class_count)

    def init_graph_label(self, label_path: str) -> None:
        '''intialize graph object by creating and storing important graph variables'''

        self.get_label_triples(label_path)

        # to store all subjects, predicates and objects
        subjects = set()
        predicates = set()
        objects = set()

        class_count = defaultdict(int)

        # loop over each triple and split 2 times on space:' '
        for triple in self.label_triples:
            # triple_list = triple[:-2].split(' ', maxsplit=2)
            triple_list = triple.split('\t', maxsplit=2)
            # skip triple if there is a blank lines in .nt files
            if triple_list != ['']:
                s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
                # add nodes and predicates
                subjects.add(s)
                predicates.add(p)
                objects.add(o)
                # check if subject is a valid entity and check if predicate is rdf:type
                if str(p) == self.RDF_TYPE.lower():
                        class_count[str(o)] += 1
                        self.node2types_dict[s].add(o)

        # create a list with all nodes and enumerate the nodes
        # nodes = list(subjects.union(objects))
        # self.enum_nodes = {node: i for i, node in enumerate(sorted(nodes))}
        # remove the rdf:type relations since we would like to predict the types
        # and enumerate the relations and save as dict
        # predicates.remove(self.RDF_TYPE.lower())
        # self.enum_relations = {rel: i for i, rel in enumerate(sorted(predicates))}
        # enumereate classes
        self.enum_classes = {lab: i for i, lab in enumerate(class_count.keys())}

        # print class occurence dict too get insight in class (im)balance
        # print(class_count)


    def print_graph_statistics(self) -> None:
        print('GRPAH STATISTICS:')
        print('number of nodes:', len(self.enum_nodes.keys()))
        print('number of edges:', len(self.graph_triples))
        print('number of relations:', len(self.enum_relations.keys()))
        print('number of classes:', len(self.enum_classes.keys()))
    
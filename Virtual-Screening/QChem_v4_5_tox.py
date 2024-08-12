import math
import os
import numpy as np
import networkx as nx
from rdkit import Chem,RDConfig
from collections import defaultdict
from rdkit.Chem import rdMolTransforms, rdMolDescriptors, AllChem
from dimod.reference.samplers import SimulatedAnnealingSampler
import dwave_networkx as dnx

#fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = AllChem.BuildFeatureFactory('./GMSFeatures.fdef')

class MolGraph: 
    '''
    *** MolGraph Class ***

    The MolGraph class provides methods for creating a labeled graph representation of a molecule. The graph is made up of nodes representing atoms and rings, and edges representing bonds between atoms and rings.

    ::: Attributes :::

    The MolGraph class has the following attributes:
    - mol: A RDKit molecule object.
    - atoms: A list of RDKit atom objects.
    - bonds: A list of RDKit bond objects.
    - rings: A list of lists of atom indices in rings.
    - num_rings: The number of rings in the molecule.
    - atom_colours: A dictionary of atom colour codes.
    - bond_colours: A dictionary of bond colour codes.
    - positions: A list of 3D coordinates for each atom in the molecule.

    ::: Methods :::
    
    The MolGraph class has the following methods:

    get_atom_labels()
        : Returns a list of atom labels for each atom in the molecule.
    get_bonds_labels()
        : Returns a list of bond labels for each bond in the molecule.
    get_atom_properties(atom)
        : Returns a dictionary of properties for a single atom in the molecule.
    get_ring_properties(mol, ring_atoms)
        : Returns a dictionary of properties for a single ring in the molecule given the atom indices in the ring.
    get_bond_properties(bond)
        : Returns a dictionary of properties for a chemical bond in the molecule.
    get_ring_pos()
        : Returns a list of 3D coordinates for the centre of each ring in the molecule.
    get_nearby_rings()
        : Returns a list of pairs of rings that are connected by a bond.
    get_nodes()
        : Returns a list of nodes and their properties, a list of node positions, and a list of node colours.
    get_edges()
        : Returns a list of edges, a list of edge colours, and a dictionary of edge types.
    get_mol_graph()
        : Returns a graph object and a list of graph properties.
    '''
    
    def __init__(self, mol, name = None):
        self.mol = mol
        self.name = name
        self.atoms = list(mol.GetAtoms())
        self.positions = mol.GetConformer(-1).GetPositions()
        self.bonds = list(mol.GetBonds())
        self.rings = [list(ring) for ring in mol.GetRingInfo().AtomRings()]
        self.num_rings = len(self.rings)
        self.pharmacophore_features = list(self.get_pharmacophore_features(self.mol))
        
        self.atom_colours = {
            'C': '#6096B4',  # Carbon
            'N': '#7286D3',  # Nitrogen
            'O': '#BB6464',  # Oxygen
            'S': '#FEBE8C',  # Sulpher
            'P': '#ADE4DB',  # Phosphorus
            'F': '#c0ff33',   # Fluorine
            'Cl': '#5B59BA',  # Chlorine
            'Br': '#551A8B',  # Bromine
            
        }
        self.bond_colours = {
            'ARTIFICIAL': '#C0C0C0',  # Artificial bond (gray, dotted line)
            'SINGLE': '#000000',  # Single bond (black)
            'DOUBLE': '#BB6464',  # Double bond (yellow)
            'TRIPLE': '#FFC800',  # Triple bond (red)
            'AROMATIC': 'blue',   # Aromatic bond
        }
    
        
    def get_atom_labels(self):
        """
        Get labels for all atoms in a molecule.
        """
        atom_labels = []
        for atom in self.atoms:
            atom_labels.append(atom.GetSymbol())
        return atom_labels
    
    def count_elements(self, group):
        counts = defaultdict(int)
        for element in group:
            counts[element] += 1
        return dict(counts)
    
    def get_bonds_labels(self):
        """
        Get labels for all bonds in a molecule.
        """
        bonds_labels = []
        for bond in self.bonds:
            order = bond.GetBondType().name
            bonds_labels.append(order)
        return bonds_labels
    
    def get_pharmacophore_features(self, mol: Chem.rdchem.Mol, feature_factory: Chem.rdMolChemicalFeatures.MolChemicalFeatureFactory = factory) -> list:
        """
        Extracts pharmacophore features for each atom in a molecule and returns a list of dictionaries containing the properties for each atom.

        The pharmacophore features considered are:

            - Acceptor -- Hydrogen Acceptor
            - Donor -- Hydrogen Donor
            - Acidic -- Anionic species
            - Basic -- Cationic species
            - Aromatic
            - Hydrophobe
            - ZnBinder -- Zinc Binder

        Args:
        ----------
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
        ----------
            pharmacophore_features: List of dictionaries containing the pharmacophore features for each atom in the molecule.

        """

        # create empty list to store dictionaries of properties for each atom


        features = feature_factory.GetFeaturesForMol(mol)
        pharmacophore_keys = ['Donor','Acceptor','ZnBinder', 'Hydrophobe','Aromatic', 'NegIonizable','PosIonizable','Toxicophore'] #change line 467 and 471 accordingly

        pharmacophore_features,pharmacophore_data =[],{}
        for key in pharmacophore_keys:
            pharmacophore_data[key]= []

        for feats in features:
            for key in pharmacophore_keys:
                if feats.GetFamily() == key:
                    pharmacophore_data[key].extend(feats.GetAtomIds())
                    
        for atom in mol.GetAtoms():
            atom_prop = {}
            for key in pharmacophore_keys:
                atom_prop[key] = False
                for value in pharmacophore_data[key]:
                    if value == atom.GetIdx():
                        atom_prop[key] = True

            atom_prop['Acidic'] = atom_prop.pop('NegIonizable') 
            atom_prop['Basic'] = atom_prop.pop('PosIonizable') 



            pharmacophore_features.append(atom_prop)

        return pharmacophore_features

    def get_atom_pair_distances(self, nodes, nodes_positions):

        '''
        Get the distance between all atom pairs.
        
        Arg:
            -------
            node (list): List of node of the graph
            node_positions (dict): Dictionary of positions keyed by nodes 

        Returns:
            -------
                distances : dictionary containing distances keyed by the atom pairs.
        '''
        distances = {(i,j): round(math.dist(nodes_positions[j], nodes_positions[i]),5) 
                        for i in nodes
                        for j in nodes
                    }
        return distances
    
    def get_atom_properties(self, atom):
        """
        Get the properties for a single atom in a molecule.
        """
        mol = atom.GetOwningMol()
        atomic_num = atom.GetAtomicNum()
        #implicit_hydrogen = atom.GetNumImplicitHs() # changed -- from GetTotalNumHs --> GetNumImplicitHs() -- Changed Implicit to Explicit Hs
        formal_charge = atom.GetFormalCharge()
        #degree = atom.GetDegree()
        bond_order = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
        total_valence = atom.GetTotalValence()
        implicit_hydrogen = int(total_valence - int(sum(bond_order))) # new way to calculate implicit Hs for atoms
        degree = atom.GetTotalNumHs() - implicit_hydrogen
        coords = list(mol.GetConformer(-1).GetAtomPosition(atom.GetIdx())) # changed -- no conformer id --> conformer id was specified to be one
        symbol = atom.GetSymbol()
        properties = {
            #'ring': False,
            'atomic_number_atom': atomic_num,
            'implicit_hydrogen': implicit_hydrogen,
            'formal_charge': formal_charge,
            'degree': int(degree),
            'bond_order': bond_order,            
            #'coords': coords,
            'symbol': symbol,
        }
        
        features = self.pharmacophore_features[atom.GetIdx()]
        
        for key, val in features.items():
            if val == True: # only adding true pharmacopore features into node properties
                properties[key] = val
            
        return properties
    
    def _get_ring_neighbors(self,ring_atoms): #changed -- Added new function to calculate bond order and degree of rings

        mol = self.mol
        ring_atoms = set(ring_atoms)
        neighboring_atoms = set()

        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            neighboring_atoms.update([neighbor.GetIdx() for neighbor in neighbors])

        unq_neighbors = list(neighboring_atoms - ring_atoms)
        unq_neighbors.sort()
        degree = len(unq_neighbors)

        bond_order = []
        for u in unq_neighbors:
            bonds = mol.GetAtomWithIdx(u).GetBonds()
            for bond in bonds:
                if bond.GetBeginAtomIdx() in ring_atoms or bond.GetEndAtomIdx() in ring_atoms:
                    bond_order.append(bond.GetBondTypeAsDouble())
        
        #bond_order = self.count_elements(bond_order) # Converting list to count dictionary

        return degree, bond_order
    

    def get_ring_properties(self, ring_atoms):
        """
        Get the properties for a single ring in a molecule given the atom indices in the ring.
        """
        mol = self.mol
        ring = Chem.PathToSubmol(mol, ring_atoms)
        atomic_nums = self.count_elements([mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring_atoms]) #changed -- set --> list

        implicit_hydrogens = int(sum([mol.GetAtomWithIdx(i).GetNumImplicitHs() for i in ring_atoms])) #changed -- set --> list -- GetTotalNumHs --> GetNumImplicitHs -- List --> Sum

        #formal_charge = sum(mol.GetAtomWithIdx(atom_idx).GetFormalCharge() for atom_idx in ring_atoms) #changed -- added total formal charge
        formal_charge = None
        degrees, bond_orders = self._get_ring_neighbors(ring_atoms)

        degree = sum([mol.GetAtomWithIdx(i).GetNumExplicitHs() for i in ring_atoms])


        coords = [list(mol.GetConformer(-1).GetAtomPosition(i)) for i in ring_atoms]  # changed -- no conformer id --> conformer id was specified to be one
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring_atoms]
        properties = {
            'ring': True,
            'ring_size': len(ring_atoms), #changed -- del 'ring_atoms': set(ring_atoms)
            'atomic_number_ring': atomic_nums,
            'implicit_hydrogen': implicit_hydrogens,
            'degree': int(degree),
            'bond_order': bond_orders,
            #'formal_charge': formal_charge,
            #'coords': coords,
            'symbol': symbols
        }
        
        features = [self.pharmacophore_features[i] for i in ring_atoms]
        
        for key in features[0].keys():
            values = [feature[key] for feature in features] # List of pharmacophore of all the atoms as True/False
            # Changed: Adding True to properties if atleast one of the atom in ring has True for given property
            if True in values: # only adding true pharmacopore features into node properties
                properties[key] = True 
            # else:
            #     properties[key] = False
            
        return properties
    
    
    def get_bond_properties(self, bond):
        """
        Calculate various properties of a chemical bond
        """
        rings = self.rings

        mol = bond.GetOwningMol()
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        e = (atom1.GetIdx(),atom2.GetIdx())

        # Calculate bond type and bond order
        # bond_type = str(bond.GetBondType())
        bond_order = bond.GetBondTypeAsDouble()

        # Calculate bond length
        # bond_length = rdMolTransforms.GetBondLength(mol.GetConformer(), atom1.GetIdx(), atom2.GetIdx())

        # Create dictionary of bond properties
        bond_properties = {
            # 'bond_type': bond_type,
            'bond_order': bond_order,
            # 'bond_length': bond_length,
        }

        return bond_properties

    def get_ring_pos(self): #need to check
        """
        Computes the centroid of each ring in the molecule 

        Args:
        ----------
            self : object
                Molecule object containing the rings and positions attributes 

        Returns:
        -------
            ring_pos : list
                A list of the centroid coordinates for each ring 
        """
        rings = self.rings
        positions = self.positions
        ring_pos = []
        for ring in rings:
            x_mean = sum([positions[i][0] for i in ring]) / len(ring)
            y_mean = sum([positions[i][1] for i in ring]) / len(ring)
            z_mean = sum([positions[i][2] for i in ring]) / len(ring)
            ring_pos.append(np.array([x_mean, y_mean, z_mean]))
        return ring_pos
    
    def get_nearby_rings(self):
        """
        Finds the rings in a molecule that are connected to each other. 

        Args:
        ----------
            self : object
                The object being passed in.

        Returns:
        -------
            ring_bonds : list
                A list of the connected rings. 
        """
        rings = self.rings
        ring_bonds = []
        for r in range(len(rings)):
            for s in range(r+1,len(rings)):
                if set(rings[r]) & set(rings[s]):
                    ring_bonds.append([r,s])         
        return ring_bonds

    def get_nodes(self):
        """
        Gets the nodes and properties of a molecule. 

        Args:
        ----------
            self : object
                The object being passed in.

        Returns:
        -------
            Nodes : list
                A list of the nodes and their properties. 
            node_pos : list
                A list of the node positions. 
            node_colours : list
                A list of the node colours. 
        """
        Nodes = []

        nodes = [atom.GetIdx() for atom in self.atoms]
        node_pos = [pos for pos in self.positions]

        for a in range(len(self.atoms)):
            atom = self.atoms[a]
            if atom.IsInRing():
                node_pos[a] = None
                nodes.remove(atom.GetIdx())
            else:
                properties = self.get_atom_properties(atom)
                Nodes.append((atom.GetIdx(), properties))

        atom_labels = self.get_atom_labels()
        node_colours = [self.atom_colours.get(str(atom_labels[n]),'#383838') for n in nodes] #milan

        ring_pos = self.get_ring_pos()
        l = len(node_pos)
        for r in range(len(self.rings)):
            ring = self.rings[r]
            nodes.append(l+r)
            properties = self.get_ring_properties(ring)
            node_pos.append(ring_pos[r])
            node_colours.append('#f5ce42') #milan
            Nodes.append((l+r, properties))

        return Nodes, node_pos, node_colours     #milan
        
    # def get_edges(self):
    #     """
    #     Generates the edges and edge properties of the molecule graph.
    
    #     Args:
    #     ----------
    #         self : MoleculeGraph
    #             The MoleculeGraph object
    
    #     Returns:
    #     -------
    #         edges : list
    #             List of edges in the molecule graph
    #         edge_colours : list
    #             List of edge colours for each edge in the molecule graph
    #         edge_type : dict
    #             Dictionary of edge types and corresponding colours for each edge in the molecule graph
    #     """
    #     edges = []
    #     edge_colours = []
    #     edge_type = {}
    #     rings = self.rings
    #     l = len(self.positions)
    #     r = 0
    #     for b in range(len(self.bonds)):
    #         bond = self.bonds[b]
    #         begin_atom = bond.GetBeginAtom()
    #         end_atom = bond.GetEndAtom()
    #         e = (begin_atom.GetIdx(), end_atom.GetIdx())

    #         if not begin_atom.IsInRing() or not end_atom.IsInRing():
    #             bond_properties = self.get_bond_properties(bond)
    #             edges.append((e[0],e[1],bond_properties))
    #             bond_type = bond.GetBondType()
    #             edge_type[str(e)] = str(bond_type)+'_'+self.bond_colours[str(bond_type)]
    #             edge_colours.append(self.bond_colours[str(bond_type)])

           

    #         ################################## not clear
    #         for r in range(len(rings)):
    #             for s in range(r+1,len(rings)):
    #                 if (e[0] in rings[r] and e[1] in rings[s]) or (e[0] in rings[s] and e[1] in rings[r]):
    #                     if not set(rings[r]) & set(rings[s]):
    #                         bond_properties = self.get_bond_properties(bond)
    #                         edges.append((e[0],e[1],bond_properties))
    #                         bond_type = bond.GetBondType()
    #                         edge_type[str(e)] = str(bond_type)+'_'+self.bond_colours[str(bond_type)]
    #                         edge_colours.append(self.bond_colours[str(bond_type)])
                
    #     for r in range(self.num_rings):
    #         ring = self.rings[r]
    #         for i in range(len(edges)):
    #             e = list(edges[i])
    #             if e[0] in ring:
    #                 e[0] = l+r
    #             elif e[1] in ring:
    #                 e[1] = l+r
    #             edges[i] = tuple(e)
        
    #     ring_bonds =  self.get_nearby_rings()
    #     for e in ring_bonds:
    #         bond_properties = {
    #             'bond_type': 'ARTIFICIAL',
    #             'bond_order': None,
    #             'bond_length': None,
    #         }
    #         edges.append((l+e[0],l+e[1],bond_properties))
    #         edge_colours.append('#f5ce42')
    #     ##########################################
            
    #     return edges, edge_colours, edge_type

    def get_edges(self):
        """
        Generates the edges and edge properties of the molecule graph.
    
        Args:
        ----------
            self : MoleculeGraph
                The MoleculeGraph object
    
        Returns:
        -------
            edges : list
                List of edges in the molecule graph
            edge_colours : list
                List of edge colours for each edge in the molecule graph
            edge_type : dict
                Dictionary of edge types and corresponding colours for each edge in the molecule graph
        """
        edges = []
        edge_colours = []
        edge_type = {}
        rings = self.rings
        l = len(self.positions)
        r = 0
        for b in range(len(self.bonds)):
            bond = self.bonds[b]
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            e = (begin_atom.GetIdx(), end_atom.GetIdx())

            if (begin_atom.IsInRing() and not end_atom.IsInRing()) or (not begin_atom.IsInRing() and end_atom.IsInRing()):
                for i in range(self.num_rings):
                    r = self.rings[i]
                    if e[0] in r:
                        bond_properties = self.get_bond_properties(bond)
                        edges.append((l + i, e[1],bond_properties))
                        bond_type = bond.GetBondType()
                        edge_type[str(e)] = str(bond_type)+'_'+self.bond_colours[str(bond_type)]
                        edge_colours.append(self.bond_colours.get(str(bond_type), '#3c4241'))
                    elif e[1] in r:
                        bond_properties = self.get_bond_properties(bond)
                        edges.append((e[0], l+i, bond_properties))
                        bond_type = bond.GetBondType()
                        edge_type[str(e)] = str(bond_type)+'_'+self.bond_colours[str(bond_type)]
                        edge_colours.append(self.bond_colours.get(str(bond_type), '#3c4241'))

            elif not begin_atom.IsInRing() and not end_atom.IsInRing():              
                bond_properties = self.get_bond_properties(bond)
                edges.append((e[0],e[1],bond_properties))
                bond_type = bond.GetBondType()
                edge_type[str(e)] = str(bond_type)+'_'+self.bond_colours[str(bond_type)]
                edge_colours.append(self.bond_colours.get(str(bond_type), '#3c4241'))


            # for the case of two rings having a bond between them
            for r in range(len(rings)):
                for s in range(r+1,len(rings)):
                    if (e[0] in rings[r] and e[1] in rings[s]) or (e[0] in rings[s] and e[1] in rings[r]):
                        if not set(rings[r]) & set(rings[s]):
                            bond_properties = self.get_bond_properties(bond)
                            edges.append((e[0],e[1],bond_properties))
                            bond_type = bond.GetBondType()
                            edge_type[str(e)] = str(bond_type)+'_'+self.bond_colours[str(bond_type)]
                            edge_colours.append(self.bond_colours[str(bond_type)])
                
        # For avoiding index mismatch    
        for r in range(self.num_rings):
            ring = self.rings[r]
            for i in range(len(edges)):
                e = list(edges[i])
                if e[0] in ring:
                    e[0] = l+r
                elif e[1] in ring:
                    e[1] = l+r
                edges[i] = tuple(e)
        
        # For adding artificial bonds
        ring_bonds =  self.get_nearby_rings()
        for e in ring_bonds:
            bond_properties = {
                'bond_type': 'ARTIFICIAL',
                'bond_order': None,
                'bond_length': None,
            }
            edges.append((l+e[0],l+e[1],bond_properties))
            edge_colours.append('#f5ce42')
            
        return edges, edge_colours, edge_type
    
    def get_mol_graph(self):
        
        """
        Returns a networkx graph object and its associated properties
        """
        
        graph = nx.Graph()
        graph.graph['name'] = self.name
        
        mol = self.mol
        Id=None
        Id = mol.GetProp('_Name')
        graph.graph['id'] = Id
        
        nodes, node_pos, node_colours = self.get_nodes() # [0], self.get_nodes()[1]  #milan
        edges, edge_colours, edge_type = self.get_edges()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        ## finding bond order and degree consistent with molecular graphs

        N = list(graph.nodes())

        for n in N:
            neighbors = list(graph.neighbors(n))
            deg = len(neighbors)
            bond_order = []
            for neighbor in neighbors:
                if graph.has_edge(n,neighbor):
                    b = graph.get_edge_data(n, neighbor)['bond_order']
                    if b is None:
                        # graph.nodes[n]['artificial'] = 'True'
                        # graph.nodes[neighbor]['artificial'] = 'True'
                        pass
                    else:
                        bond_order.append(b)
            

            #graph.nodes[n]['degree'] += int(deg)# with explicit Hs
            graph.nodes[n]['degree'] = int(deg) # without explicit Hs
            graph.nodes[n]['bond_order'] = self.count_elements(bond_order)
            #graph.nodes[n]['bond_order'] = bond_order 

        graph_properties = [node_pos, node_colours, edge_colours, nodes, edges, edge_type]  #milan
        graph_properties[0] = [n if n is None else n[:-1] for n in graph_properties[0]]
        
        node_idx = [n[0] for n in nodes]
        distances = self.get_atom_pair_distances(node_idx, node_pos)
        return graph, graph_properties, distances
    





class ConflictGraph:
    """
    A class that creates the conflict graph for two molecular graphs G1 and G2, where each node in the conflict graph represents a pair of nodes from G1 and G2 that have matching values for all critical labels, and each edge in the conflict graph represents a conflict between two pairs of nodes.
    """

    def __init__(self, G1, G2, distances_G1, distances_G2, label_weights=None, critical_labels=None, off_labels=None):
        """
        Initializes the conflict graph object.

        Args:
        ----------
            G1 (networkx.Graph): the first molecular graph
            G2 (networkx.Graph): the second molecular graph
            distances_G1: Atom pair distances for first molecular grpah
            distances_G2: Atom pair distances for second molecular grpah
            label_weights (dict): a dictionary of weights for different labels
            critical_labels (set): a set of labels that must match exactly for nodes to be considered as potential matches
            off_labels (set): a set of labels to be excluded from the calculation of edge weights
        """

        # Set default values for the arguments
        if label_weights is None:
            label_weights = {'ring': 0,'ring_size': 5, 'ring_atoms': 1, 'atomic_number_atom': 1, 'atomic_number_ring': 1, 'implicit_hydrogen': 1, 'formal_charge': 1,
                             'degree': 1, 'bond_order': 1, 'distances': 1, 'coords': 1, 'symbol': 1,'Donor':1, 'Acceptor': 1, 'Aromatic': 1, 'Hydrophobe':1, 'ZnBinder': 1, 'Acidic': 1, 'Basic': 1}
        
        if critical_labels is None:
            critical_labels = {'atomic_number_atom','Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'ZnBinder','Acidic','Basic'}
            
        if off_labels is None:
            off_labels = {'ring_atoms', 'coords', 'symbol'}
        
        self.G1 = G1
        self.G2 = G2
        self.distances_G1= distances_G1
        self.distances_G2= distances_G2
        self.label_weights = label_weights
        self.critical_labels = critical_labels
        self.off_labels = off_labels
        self.Vc = None
        self.Ec = None

    def get_potential_matches(self):
        """
        Returns a list of all pairs of nodes in G1 and G2 that have matching values for all critical labels.

        Args:
        ----------

        Returns:
        ----------
            Vc (list): a list of all pairs of nodes in G1 and G2 that have matching values for at least one critical label
            matched_labels (list of lists): a list of lists containg all the matched critical labels for each node in Vc
        """

        G1 = self.G1
        G2 = self.G2
        
        # create a list of node pairs that represent potential matches and matched labels
        Vc_K = [(u, v, [key for key in self.critical_labels
                      # check that nodes have the critical label and match on that label
                      if key in G1.nodes[u] and key in G2.nodes[v] and G1.nodes[u][key] == G2.nodes[v][key] ]) # removed-- condition for removing False labelled properties
                # include only those (u,v) for which atleast one of the critical label match
                for u in G1.nodes() for v in G2.nodes() if (any(key for key in self.critical_labels
                                                               if key in G1.nodes[u] and key in G2.nodes[v] and G1.nodes[u][key] == G2.nodes[v][key])) ## removed-- condition for removing False labelled properties
                                                               and ( G1.nodes[u].get('ring') == G2.nodes[v].get('ring'))]#and (G1.nodes[u].get('artificial')== G2.nodes[v].get('artificial'))]
        
        
        # Seprate the nodes and matched critical labels
        Vc = [] # contains nodes of the conflict graph (each element is a tuple)
        matched_labels = [] # contains matched keys for each node (each element is a list)
        for u,v,k in Vc_K:
            Vc.append((u,v))
            matched_labels.append(k)
        
        return Vc, matched_labels

    def get_node_weights(self, Vc, matched_labels):
        """
        Calculates the weight of each node in the conflict graph.

        Args:
        ----------:float
            Vc (list): a list of all pairs of nodes in G1 and G2 that have matching values for at least one critical label
            matched_labels (list of lists): a list of lists containg all the matched critical labels for each node in Vc

        Returns:
        ----------
            weights (dict): A dictionary of nodes weights, keyed by pairs of nodes in Vc.
        """

        G1 = self.G1
        G2 = self.G2
        
        # Calculate the weight of each node in the conflict graph
        weights = {}
        for n, labels in zip(Vc,matched_labels):
            u, v = n
            # The weight of an node is initially equal to sum of the weights of matched critical labels
            weight = sum([self.label_weights[label] for label in labels])
            # Checking for non-critical labels
            for key in set(G1.nodes[u]) - self.critical_labels:
                if key in self.off_labels:
                    # Ignore this label if it is in the set of off-labels
                    continue
                elif key in G2.nodes[v] and G1.nodes[u][key] == G2.nodes[v][key]:
                    # Increase the weight of the node by weight of the label
                    weight += self.label_weights[key]
            weights[(u,v)] = weight
            
             #features of ring are given weight 5 
#             if 'atomic_number_ring' in list(G1.nodes[u].keys()) and 'atomic_number_ring' in list(G2.nodes[v].keys()):    
#                  weights[(u,v)] = weight+5
#             else::float
#                  weights[(u,v)] = weight+1
            
        return weights
    
    def get_conflict_edges(self, Vc, dt):
        '''
        Returns a list of edges in the conflict graph given a list of critical nodes.

        Args:
        ----------
            Vc (list): A list of tuples, where each tuple represents a critical node in the form (u, v), where u is a node in G1 and v is a node in G2.
            dt (float): A threshold for distance conflict
        Returns:
        ----------
            Ec (list): A list of tuples representing edges in the conflict graph, where each tuple contains two tuples representing the corresponding critical nodes in G1 and G2.
        '''
        G1 = self.G1
        G2 = self.G2
        
        distances_G1 = self.distances_G1
        distances_G2 = self.distances_G2

        # Initialize an empty list to store the edges of the conflict graph
        Ec = []

        # Iterate through all pairs of critical nodes
        for i in range(len(Vc)):
            for j in range(i+1, len(Vc)):
                u1, u2 = Vc[i]
                v1, v2 = Vc[j]

                # If either of the critical nodes in G1 or G2 are the same,
                # then create an edge between their corresponding nodes in the conflict graph
                if u1 == v1 or u2 == v2:
                    Ec.append(((u1, u2), (v1, v2)))

                # If the corresponding nodes in G1 and G2 are different, create an edge between them
                # elif u1 != v1 or u2 != v2:
                #     # If an edge exists in G1 but not G2, or vice versa, create an edge between the corresponding nodes in the conflict graph
                #     if (G1.has_edge(u1, v1) and not G2.has_edge(u2, v2)) or (not G1.has_edge(u1, v1) and G2.has_edge(u2, v2)):
                #         Ec.append(((u1, u2), (v1, v2)))

                    # # If an edge exists in both G1 and G2, but with different attributes, create an edge between the corresponding nodes in the conflict graph
                    # if (G1.has_edge(u1, v1) and G2.has_edge(u2, v2) and G1.edges[u1, v1] != G2.edges[u2, v2]):
                    #     Ec.append(((u1, u2), (v1, v2)))
                
                # distance mismatch
                if (u1,v1) in list(distances_G1.keys()) and (u2,v2) in list(distances_G2.keys()):
                    G1d = distances_G1[(u1,v1)]
                    G2d = distances_G2[(u2,v2)]
                    if abs(G1d-G2d) > dt:
                        Ec.append(((u1, u2), (v1, v2)))

        return Ec
    
    def create_conflict_graph(self, dt = 0):
        '''
        Creates a conflict graph using the two input graphs G1 and G2.

        Returns:
        ----------
            Gc (NetworkX graph object): A graph object representing the conflict graph.
        '''
        
        Vc, matched_labels = self.get_potential_matches()  # Get nodes of the conflict graph       
        self.Vc = Vc
        
        Ec = self.get_conflict_edges(Vc, dt) # Get edges of the conflict graph
        self.Ec = Ec
        
        weights = self.get_node_weights(Vc,matched_labels) # Get weights of the nodes
        
        # Create a new graph to represent the conflict graph, and add nodes and edges to it. 
        Gc = nx.Graph()
        Gc.add_nodes_from(Vc)
        Gc.add_edges_from(Ec)
        
        # Add weight attribute to the nodes
        nx.set_node_attributes(Gc, weights, name="weight")
        
        # Return the conflict graph
        return Gc
    
    def get_MWIS(self,sampler = SimulatedAnnealingSampler(), dt:float =1.5):
    
        '''
        Finds the Maximum Weighted Independent Set from the Conflict Graph object.

        Args:
        ----------
            Sampler (dimod.reference.samplers / dwave.system.samplers / dwave.system.composites): Samplers provided by Dimod or Dwave.
            dt (float): Threshold to the distance conflict. The default value is set to 1.5.


        Returns:
        -----------
            mwis (dict): A dictionary where keys gives the nodes in MWIS and values give the corresponding weight.
        '''

        Gc = self.create_conflict_graph(dt=dt)
        MWI_set = dnx.maximum_weighted_independent_set(Gc,"weight",sampler)

        weight_set = [Gc.nodes[node]["weight"] for node in MWI_set]
        #total_weight = sum(weight_set)
        mwis =  {key: value for key, value in zip(MWI_set, weight_set)}
        return mwis
    
    def _feat_set_calc(self,G):
        '''
        Finds the features present in the Graph.

        Args:
        ----------
            G (NetworkX graph object): A graph object representing the Molecule.
        
        Returns:
        -----------
            feat_dict (dict): A dictionary where key is the features in the graph and value is number of nodes where feature is present

        '''

        label_weights = self.label_weights
        off_labels = self.off_labels

        feat_dict = {key: 0 for key in label_weights.keys()}
        for node in G.nodes():
            node_labels = set(G.nodes[node].keys()) - off_labels
            for key in node_labels:
                if not (G.nodes[node][key] == False and str(G.nodes[node][key]) !=str(0)):
                    feat_dict[key] += 1
        return feat_dict
    
    def get_weighted_sim_score(self, sampler = SimulatedAnnealingSampler(), dt:float = 1.5, delta:float = 0.5):

        '''
        Finds the weighted similarity score of two molecules.

        Args:
        ----------
            Sampler (dimod.reference.samplers / dwave.system.samplers / dwave.system.composites): Samplers provided by Dimod or Dwave.
            dt (float): Threshold to the distance conflict. The default value is set to 1.5.
            delta (float): Parameter that used in the similarity measure calculation, takes value between [0,1]. Default value set to be 0.5.
        
        Returns:
        -----------
            sim_score (float): A float value representing the similarity between two molecules forming the conflict graph.

        '''
        

        G1 = self.G1
        G2 = self.G2
        Gc = self.create_conflict_graph(dt)
        label_weights = self.label_weights
        critical_labels = self.critical_labels
        off_labels = self.off_labels
        temp = self.get_MWIS(sampler,dt)
        MWI_set = list(temp.keys())
        #MWI_set = list(self.get_MWIS(sampler,dt).keys())
        #print(temp)

        FA = self._feat_set_calc(G=G1)
        FB = self._feat_set_calc(G=G2)
        alpha_den = beta_den = 0
        for key in label_weights.keys():
            alpha_den+=FA[key]*label_weights[key]
            beta_den+=FB[key]*label_weights[key]
        num=sum([Gc.nodes[node]["weight"] for node in MWI_set])

        alpha=num/alpha_den
        beta=num/beta_den
        sim_score=delta*min(alpha,beta)+(1-delta)*max(alpha,beta)

        return sim_score
    
    def get_sim_score(self, sampler = SimulatedAnnealingSampler(), dt:float = 1.5, delta:float = 0.5):

        '''
        Finds the similarity score of two molecules.

        Args:
        ----------
            Sampler (dimod.reference.samplers / dwave.system.samplers / dwave.system.composites): Samplers provided by Dimod or Dwave.
            dt (float): Threshold to the distance conflict. The default value is set to 1.5.
            delta (float): Parameter that used in the similarity measure calculation, takes value between [0,1]. Default value set to be 0.5.
        
        Returns:
        -----------
            sim_score (float): A float value representing the similarity between two molecules forming the conflict graph.

        '''

        G1 = self.G1
        G2 = self.G2
        temp = self.get_MWIS(sampler,dt)
        MWI_set = list(temp.keys())

        v1 = len(G1.nodes())
        v2 = len(G2.nodes())
        vc = len(MWI_set)        # assume bijection criteria to be satisfied

        alpha= vc/v1
        beta= vc/v2
        sim_score=delta*min(alpha,beta)+(1-delta)*max(alpha,beta)

        return sim_score







    


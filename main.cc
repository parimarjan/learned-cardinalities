#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <emmintrin.h>
#include <map>
#include <unordered_map>
#include <math.h> 
#include <set>


using namespace std;


struct Edges
{
	int col_num,row_num;
	vector<vector<double> > prob_matrix;
	vector<double> col_sum;
	vector<double> row_sum;

	void init(int size1,vector<int> &data_array1,int size2,vector<int> &data_array2,vector<int> &count_column)
	{
		prob_matrix.resize(size1);
		for(int i=0;i<size1;i++)
		{
			prob_matrix[i].resize(size2,0.0);
		}

		col_sum.resize(size1);
		row_sum.resize(size2);

		col_num=size1;
		row_num=size2;

		double total_count=0.0;
		int row_count=count_column.size();

		for(int i=0;i<row_count;i++)
		{
			prob_matrix[data_array1[i]][data_array2[i]]+=count_column[i];
			total_count+=count_column[i];
		}	

		for(int i=0;i<size1;i++)
		{
			double sum=0.0;
			for(int j=0;j<size2;j++)
			{
				prob_matrix[i][j]/=total_count;
				sum+=prob_matrix[i][j];
			}
			col_sum[i]=sum;
		}

		for(int j=0;j<size2;j++)
		{
			double sum=0.0;
			for(int i=0;i<size1;i++)
			{
				sum+=prob_matrix[i][j];
			}
			row_sum[j]=sum;
		}
	}

	double cal_mutual_info()
	{
		double mutal_info=0.0;
		for(int i=0;i<col_num;i++)
		{
			for(int j=0;j<row_num;j++)
			{
				if(col_sum[i]!=0.0 && row_sum[j]!=0.0)
				{
					mutal_info+=prob_matrix[i][j]*(log(prob_matrix[i][j]/(col_sum[i]*row_sum[j])));
				}
			}
		}

		return mutal_info;
	}

	void print()
	{
		cout<<"number of columns are:" <<col_num<<" : rows are "<<row_num<<endl;

		for(int i=0;i<col_num;i++)
		{
			cout<<col_sum[i]<<" ";
		}

		cout<<endl;

		for(int i=0;i<col_num;i++)
		{
			cout<<row_sum[i]<<" ";
			for(int j=0;j<row_num;j++)
			{
				cout<<prob_matrix[i][j]<<" ";
			}
			cout<<endl;
		}
	}



};

struct Nodes
{
	int alphabet_size;
	int parent_ptr;
	vector<double> prob_list;	
	vector<int> child_ptr;

	void init(vector<int> &data_array,vector<int> &count_column)
	{
		map<int,int> alphabet_map;
		int row_num=count_column.size();
		double total_count=0;

		for(int i=0;i<row_num;i++)
		{
			total_count+=count_column[i];

			if(alphabet_map.find(data_array[i])==alphabet_map.end())
			{
				alphabet_map[data_array[i]]=count_column[i];
			}
			else
			{
				alphabet_map[data_array[i]]+=count_column[i];
			}
		}

		alphabet_size=alphabet_map.size();

		for (std::map<int,int>::iterator it=alphabet_map.begin(); it!=alphabet_map.end(); ++it)
		{
			prob_list.push_back(it->second*1.00/total_count);
		}
	}

	void print()
	{
		cout<<"alphabet_size is: "<<alphabet_size<<endl;
		cout<<"parent_ptr is: "<<parent_ptr<<endl;

		cout<<"children are:";

		for(int i=0;i<child_ptr.size();i++)
		{
			cout<<child_ptr[i]<<" ";
		}
		cout<<endl;

		cout<<"probablity list: ";

		for(int i=0;i<alphabet_size;i++)
		{
			cout<<prob_list[i]<<" ";
		}
		cout<<endl;
	}


};


struct mst_sort{
int a,b;
double val;
};

bool sortFunc(mst_sort &a,mst_sort &b)
{
	return a.val>b.val;
}

struct Graphical_Model
{
	int root;
	int graph_size;
	vector<Nodes> node_list;
	vector<vector<Edges> > edge_matrix;

	void fill_data(vector<vector<int> > &data_matrix, vector<int> &count_column)
	{
		int col_num=data_matrix.size();
		int row_num=count_column.size();

		graph_size=col_num;

		for(int i=0;i<col_num;i++)
		{
			Nodes temp;
			temp.init(data_matrix[i],count_column);
			node_list.push_back(temp);
		}

		edge_matrix.resize(col_num);

		for(int i=0;i<col_num;i++)
		{
			for(int j=i+1;j<col_num;j++)
			{
				Edges temp_edge;
				temp_edge.init(node_list[i].alphabet_size,data_matrix[i],node_list[j].alphabet_size,data_matrix[j],count_column);
				edge_matrix[i].push_back(temp_edge);
			}
		}

	}

	void init(vector<vector<int> > &data_matrix, vector<int> &count_column)
	{
		fill_data(data_matrix,count_column);
	}

	void create_graph(vector<mst_sort> &added_edges)
	{
		vector<int> process_order;
		root=added_edges[0].a;
		node_list[root].parent_ptr=-1;
		process_order.push_back(root);

		while(process_order.size()!=0)
		{
			int curr_node=process_order[0];

			for(int i=0;i<added_edges.size();i++)
			{
				if(added_edges[i].a==curr_node && added_edges[i].b!=node_list[curr_node].parent_ptr)
				{
					node_list[added_edges[i].b].parent_ptr=curr_node;
					node_list[curr_node].child_ptr.push_back(added_edges[i].b);
					process_order.push_back(added_edges[i].b);
				}

				if(added_edges[i].b==curr_node && added_edges[i].a!=node_list[curr_node].parent_ptr)
				{
					node_list[added_edges[i].a].parent_ptr=curr_node;
					node_list[curr_node].child_ptr.push_back(added_edges[i].a);
					process_order.push_back(added_edges[i].a);
				}
			}

			process_order.erase(process_order.begin());

		}

	}

	void MST()
	{
		vector<mst_sort> mutual_info_vec;

		set<int> vertices_added;
		set<int>::iterator it1,it2;

		vector<mst_sort> added_edges;


		for(int i=0;i<graph_size;i++)
		{
			for(int j=i+1;j<graph_size;j++)
			{
				mst_sort temp;
				temp.a=i;
				temp.b=j;
				temp.val=edge_matrix[i][j-i-1].cal_mutual_info();
				mutual_info_vec.push_back(temp);
			}
		}

		std::sort(mutual_info_vec.begin(),mutual_info_vec.end(),sortFunc);

		int vec_size=mutual_info_vec.size();
	
		for(int i=0;i<mutual_info_vec.size();i++)
		{
			it1=vertices_added.find(mutual_info_vec[i].a);
			it2=vertices_added.find(mutual_info_vec[i].b);
			if(!(it1!=vertices_added.end() && it2!=vertices_added.end()))
			{
				added_edges.push_back(mutual_info_vec[i]);
				vertices_added.insert(mutual_info_vec[i].a);
				vertices_added.insert(mutual_info_vec[i].b);
			}

			if(vertices_added.size()==graph_size)
			{
				break;
			}
		}

		create_graph(added_edges);

	}

	void train()
	{
		MST();
	}

	double dfs(int curr_node,int par_val,vector<set<int> > &filter,bool approx,double frac)
	{
		int parent_ptr=node_list[curr_node].parent_ptr;
		Edges* temp_edge;

		if(curr_node<parent_ptr)
		{
			temp_edge=&edge_matrix[curr_node][parent_ptr-1-curr_node];	
		}
		else
		{
			temp_edge=&edge_matrix[parent_ptr][curr_node-1-parent_ptr];
		}
		
		double ans=0.0;
		int alp_size=node_list[curr_node].alphabet_size;


		int filter_size=filter[curr_node].size();
		int actual_eval=filter_size*frac;
		set<int> new_filter;
		set<int>::iterator iter=filter[curr_node].begin();


		if(approx)
		{	int count=0;
			while(true)
			{
				int ind=rand()%filter_size;
				advance(iter,ind);
				ind= *(iter);
				iter=filter[curr_node].begin();

				if(new_filter.find(ind)==new_filter.end())
				{
					count++;
					new_filter.insert(ind);
				}

				if(count>=actual_eval)
				{
					break;
				}
			}
		}
		else
		{
			new_filter=filter[curr_node];
		}

		for(int i=0;i<alp_size;i++)
		{
			if(new_filter.find(i)==new_filter.end())
			{
				continue;
			}

			double children=1.0;
			for(int j=0;j<node_list[curr_node].child_ptr.size();j++)
			{
				children*=dfs(node_list[curr_node].child_ptr[j],i,filter,approx,frac);
			}

			children/=node_list[curr_node].prob_list[i];

			if(curr_node<parent_ptr)
			{
				ans+=temp_edge->prob_matrix[i][par_val]*children;
			}
			else
			{
				ans+=temp_edge->prob_matrix[par_val][i]*children;
			}
		}

		ans*=filter_size;
		ans/=actual_eval;

		return ans;

	} 


	double eval(vector<set<int>  > &filter,bool approx,double frac)
	{
		double ans=0.0;
		int curr_node=root;
		int alp_size=node_list[curr_node].alphabet_size;

		int filter_size=filter[curr_node].size();
		int actual_eval=filter_size*frac;
		set<int> new_filter;
		set<int>::iterator iter=filter[curr_node].begin();

		

		if(approx)
		{	int count=0;
			while(true)
			{
				int ind=rand()%filter_size;
				advance(iter,ind);
				ind= *(iter);
				iter=filter[curr_node].begin();

				if(new_filter.find(ind)==new_filter.end())
				{
					count++;
					new_filter.insert(ind);
				}

				if(count>=actual_eval)
				{
					break;
				}
			}
		}
		else
		{
			new_filter=filter[curr_node];
		}

		

		for(int i=0;i<alp_size;i++)
		{

			if(new_filter.find(i)==new_filter.end())
			{
				continue;
			}

			double children=1.0;
			for(int j=0;j<node_list[curr_node].child_ptr.size();j++)
			{
				children*=dfs(node_list[curr_node].child_ptr[j],i,filter,approx,frac);
			}

			children*=node_list[curr_node].prob_list[i];

			ans+=children;

			// if(curr_node<parent_ptr)
			// {
			// 	ans+=temp_edge.prob_matrix[i][par_val]*children;
			// }
			// else
			// {
			// 	ans+=temp_edge.prob_matrix[par_val][i]*children;
			// }
		}

		ans*=filter_size;
		ans/=actual_eval;

		return ans;
	}

	void print()
	{
		cout<<"PRINTING GRAPH"<<endl;

		cout<<"graph size is: "<<graph_size<<endl; 

		for(int i=0;i<node_list.size();i++)
		{
			node_list[i].print();
		}

		for(int i=0;i<edge_matrix.size();i++)
		{
			for(int j=0;j<edge_matrix[i].size();j++)
			{
				cout<<"edge btw: "<<i<<" "<<i+1+j<<endl;
				edge_matrix[i][j].print();
			}

		}
	}



};

Graphical_Model pgm;

void init(vector<vector<int> > &data_matrix,vector<int> &count_column)
{
	pgm.init(data_matrix,count_column);
}

void train()
{
	pgm.train();
}

double eval(vector<set<int>  > &filter,bool approx,double frac)
{
	return pgm.eval(filter,approx,frac);
}



int main(int argc, char *argv[])
{

	vector<vector<int> > data_vec(3);
	vector<int> count_column;
	vector<set<int> > vec_set(3);

	for(int i=0;i<1000;i++)
	{
		count_column.push_back(1);
		data_vec[0].push_back(i);
		data_vec[1].push_back(i);
		data_vec[2].push_back(i);
		vec_set[0].insert(i%500);
		vec_set[1].insert(i%500);
		vec_set[2].insert(i%500);
	} 

	init(data_vec,count_column);
	// pgm.print();

	train();
	// pgm.print();

	cout<<eval(vec_set,true,0.4)<<" : is the probablity"<<endl;

	return 0;
}

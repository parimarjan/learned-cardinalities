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
#include <map>
#include <fstream>


using namespace std::chrono;

using namespace std;

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
		double epsilon=0.0001/(1.000*col_num*row_num);
		for(int i=0;i<col_num;i++)
		{
			for(int j=0;j<row_num;j++)
			{
				if(col_sum[i] > epsilon && row_sum[j]>epsilon && prob_matrix[i][j]>epsilon)
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
				cout<<temp.val<<" : mutual info "<<i<<" "<<j<<endl;
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

	map<int ,double> eval(int curr_node,vector<set<int>  > &filter,bool approx,double frac)
	{
		double ans=0.0;
		int alp_size=node_list[curr_node].alphabet_size;

		int filter_size=filter[curr_node].size();
		int actual_eval=filter_size*frac;
		set<int> new_filter;
		set<int>::iterator iter=filter[curr_node].begin();

		map<int,double> curr_vals;

		for(std::set<int>::iterator it=filter[curr_node].begin();it!=filter[curr_node].end();it++)
		{
			curr_vals[*it]=1.0;
		}

		vector<int> curr_child=node_list[curr_node].child_ptr;

		for(int i=0;i<curr_child.size();i++)
		{
			int child_node=curr_child[i];
			map<int,double> child_map=eval(curr_child[i],filter,approx,frac);

			Edges *temp_edge;

			if(curr_node<child_node)
			{
				temp_edge=&edge_matrix[curr_node][child_node-curr_node-1];
			}
			else
			{
				temp_edge=&edge_matrix[child_node][curr_node-child_node-1];
			}

			for(std::map<int,double>::iterator it_par=curr_vals.begin();it_par!=curr_vals.end();it_par++)
			{
				double ans=0.0;
				int par_val=it_par->first;
				double prob_par=node_list[curr_node].prob_list[par_val];
				for(std::map<int,double>::iterator it_kid=child_map.begin();it_kid!=child_map.end();it_kid++)
				{
					int child_val=it_kid->first;
					if(child_val<par_val)
					{
						ans+=(temp_edge->prob_matrix[child_val][par_val]*it_kid->second)/prob_par;
					}
					else
					{
						ans+=(temp_edge->prob_matrix[par_val][child_val]*it_kid->second)/prob_par;
					}

				}

				if(approx)
				{
					ans/=frac;
				}

				curr_vals[par_val]*=ans;
			}
		}

		return curr_vals;
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
	double ans=0.0;
	map<int,double> root_map;
	vector<set<int> > new_filter(filter.size());

	if(approx)
	{
		for(int i=0;i<filter.size();i++)
		{
			int count=frac*filter[i].size();
			vector<int> v(filter[i].begin(),filter[i].end());
			std::random_shuffle(v.begin(),v.end());

			for(int j=0;j<count;j++)
			{
				new_filter[i].insert(v[j]);
			}

		}

		root_map=pgm.eval(pgm.root,new_filter,approx,frac);
	}
	else
	{
		root_map=pgm.eval(pgm.root,filter,approx,frac);
	}


	for(std::map<int,double>::iterator it=root_map.begin();it!=root_map.end();it++)
	{
		ans+=it->second*pgm.node_list[pgm.root].prob_list[it->first];
	}

	if(approx)
	{
		ans/=frac;
	}

	return ans;

}

extern "C" void test(int test)
{
  cout << "hello! " << endl;
  cout << test << endl;
}

extern "C" void py_init(int *data, int row_sz, int col_sz,int *count_ptr,int dim_col)
{
  vector<vector<int> > data_matrix(col_sz);
  //cout << "row: " << endl;
  //cout << row_sz << endl;
  //cout << "col: " << endl;
  //cout << col_sz << endl;

  for(int i=0;i<row_sz;i++)
  {
  	for(int j=0;j<col_sz;j++)
  	{
  		data_matrix[j].push_back(*data);
  		data++;
  	}
  }
  //cout << "reading in data done" << endl;

  vector<int> count_column;

  for(int i=0;i<dim_col;i++)
  {
  	count_column.push_back(*count_ptr);
  	count_ptr++;
  }
  init(data_matrix,count_column);

  return ;
}

extern "C" void py_train()
{
  train();
  return;
}

extern "C" double py_eval(int **data, int *lens,int n_ar,int approx,double frac)
{
  vector<set<int> > filter(n_ar);

  for(int i=0;i<n_ar;i++)
  {
  	int *ans=data[i];
  	for(int j=0;j<lens[i];j++)
  	{
  		filter[i].insert(*ans);
  		ans++;
  	}
  }
  bool app=false;
  if(approx!=0)
  {
    app=true;
  }
  //cout << "app: " << app << endl;
  //cout << "frac: " << frac << endl;

  double ans= eval(filter,app,frac);

  return ans;
}


extern "C" double test_inference(int **ar, int *lens, int n_ar)
{
	cout << "test inference!" << endl;
	int ii,jj,kk;
	for (ii = 0; ii<n_ar;ii++){
			for (jj=0;jj<lens[ii];jj++){
					printf("%d\t",ar[ii][jj]);
			}
			printf("\n");
			fflush(stdout);
	}
  return 4.0;
}

int main(int argc, char *argv[])
{

	fstream file,file1; 

	string a,b,c;
	double p,q,r;

	vector<vector<int> > data_vec(3);
	vector<int> count_column;
	vector<set<int> > vec_set(3);

  
    // Open an existing file 
    file.open("data.csv", ios::in); 

    while (getline(file, a, ',')) {
    p=atof(a.c_str());
  
    getline(file, b, ',') ;
    q=atof(b.c_str());
    
    getline(file, c);
    r=atof(c.c_str());
    
    data_vec[0].push_back(p);
    data_vec[1].push_back(q);
    data_vec[2].push_back(r);

    // cout<<p<<","<<q<<","<<r<<endl;

	}

	file1.open("counts.csv", ios::in); 


	while (getline(file1, a)) {
    p=atof(a.c_str());   
    count_column.push_back(p);

	}



	int myints1[]= {7,6,1,11,10,9,8};
	std::set<int> MySet1(myints1, myints1 + 7);
	vec_set[0]=MySet1;

	int myints2[]= {10,4};
	std::set<int> MySet2(myints1, myints1 + 2);
	vec_set[1]=MySet2;

	int myints3[]= {4,6,8,9,10,5,7};
	std::set<int> MySet3(myints1, myints1 + 7);
	vec_set[2]=MySet3;

	// data_vec[0].push_back(0);
	// data_vec[0].push_back(1);
	// data_vec[0].push_back(2);
	// data_vec[1].push_back(0);
	// data_vec[1].push_back(1);
	// data_vec[1].push_back(2);

	// count_column.push_back(100);
	// count_column.push_back(20);
	// count_column.push_back(30);

	// vec_set[0].insert(0);
	// vec_set[0].insert(1);
	// vec_set[0].insert(2);
	// vec_set[1].insert(1);
	// vec_set[1].insert(2);

	// for(int i=0;i<10000;i++)
	// {
	// 	count_column.push_back(1);
	// 	data_vec[0].push_back(i);
	// 	data_vec[1].push_back(i);
	// 	data_vec[2].push_back(i);
	// 	vec_set[0].insert(i%5000);
	// 	vec_set[1].insert(i%5000);
	// 	vec_set[2].insert(i%5000);
	// }

	for(int i=0;i<data_vec.size();i++)
	{
		for(int j=0;j<data_vec[i].size();j++)
		{
			cout<<data_vec[i][j]<<" ";
		}
		cout<<endl;
	}

	for(int i=0;i<count_column.size();i++)
	{
		cout<<count_column[i]<<" ";
	}



	init(data_vec,count_column);
	pgm.print();

	train();
	pgm.print();

	high_resolution_clock::time_point start_time, end_time;

	start_time = high_resolution_clock::now();

	cout<<eval(vec_set,false,1.0)<<" : is the probablity"<<endl;

	end_time = high_resolution_clock::now();
    cout<<duration_cast < duration < float > > (end_time - start_time).count()<<" eval time  "<<endl<<endl;

	return 0;
}

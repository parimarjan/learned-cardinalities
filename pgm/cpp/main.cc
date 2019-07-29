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
//#include <assert>
#include <cassert>
#include <Eigen/Dense>
//#include <Eigen/Eigen>

//using namespace Eigen;
using namespace std::chrono;
using namespace std;

bool VERBOSE = false;

// in the future, we may want to experiment with other stuff here
// 1 ==> store just the joint probability in the SVD matrix
// 2 ==> store the difference from the indepence assumption
int SVD_VERSION = 1;

/* assumes data is stored in column-major order */
Eigen::MatrixXd ConvertToEigenMatrix(std::vector<std::vector<double>> data)
{
    Eigen::MatrixXd eMatrix(data[0].size(), data.size());

    for (int i = 0; i < data.size(); ++i)
        //eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
        eMatrix.col(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
    return eMatrix;
}

// FIXME: temporary till assert can be included
void _ASSERT(int val1, int val2) {
  if (val1 != val2) {
    cout << "assertion failed" << endl;
    exit(-1);
  }
}

double elem_diff(vector<double> vec1, vector<double> vec2)
{
 double diff = 0.00;
 for (int i = 0; i < vec1.size(); i++) {
    diff += abs(vec1[i] - vec2[i]);
 }
 return diff;
}

bool all_close(vector<double> vec1, vector<double> vec2, double eps)
{
 double diff = 0.00;
 for (int i = 0; i < vec1.size(); i++) {
    diff += abs(vec1[i] - vec2[i]);
 }
 if (diff > eps) {
   return false;
 }
 return true;
}

/* Converts to std::vec<vec<..>> in column major form.
 *
 * @mat: Eigen format. TODO: should be able to take any Eigen matrix type.
 * @num_cols: -1 = all columns.
 */
vector<vector<double>> ConvertEigenToVec(Eigen::MatrixXd mat, int num_cols)
{
  //vector< vector<double> > vec(mat.cols(), vector<double>(mat.rows(), 0));
	vector<vector<double>> vec;
  if (num_cols == -1) num_cols = mat.cols();
  vec.resize(num_cols);
  for (int i = 0; i < num_cols; i++) {
    // initialize the column from eigen matrix
    vec[i].resize(mat.rows(),0.00);
    double *start = mat.data() + i*mat.rows();
    std::copy(start, start + mat.rows(), vec[i].begin());
  }

  _ASSERT(vec.size(), mat.cols());
  _ASSERT(vec[0].size(), mat.rows());
  return vec;
}

struct Edges
{
	int col_num,row_num;
  // stored in column major form. Outer vector represents columns, then each
  // contiguous inner vector represents a single column of data.
  // each element, (i,j) represents the joint probability of x_i, x_j.
  // FIXME: indices i,j are based on the node positions assigned at the start
  // (?)
	vector<vector<double> > prob_matrix;
	vector<double> col_sum;
	vector<double> row_sum;

  /* @size1: num columns
   * @size2: num rows
   */
  void init(int size1,vector<int> &data_array1,int size2,vector<int>
      &data_array2,vector<int> &count_column, bool use_svd, int
      num_singular_vals)
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

  void update_joint_svd_edge(vector<double> prob1, vector<double> prob2,
      int num_singular_vals)
  {
    // prob matrix has been appropriately initialized, and should not be
    // changed again. If svd flag is set, we will replace it with a SVD
    // reconstruction, based on top-k singular values

    if (SVD_VERSION == 2) {
      for (int i = 0; i < prob_matrix.size(); i++) {
        for (int j = 0; j < prob_matrix[0].size(); j++) {
          // storing only the difference from the independence assumption
          cout << "orig: " << prob_matrix[i][j] << endl;
          this->prob_matrix[i][j] = prob_matrix[i][j] - (prob1[i]*prob2[j]);
          cout << "updated: " << prob_matrix[i][j] << endl;
        }
      }
    }

    // will replace prob_matrix with the svd-d' version

    Eigen::MatrixXd prob_eigen = ConvertToEigenMatrix(prob_matrix);
    // TODO: can we avoid materializing the whole matrices?
    Eigen::BDCSVD<Eigen::MatrixXd> svd(prob_eigen, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd &U = svd.matrixU();
    const Eigen::MatrixXd &V = svd.matrixV();
    const Eigen::VectorXd &SVec = svd.singularValues();

    if (num_singular_vals == -1) num_singular_vals = SVec.size();
    // else, only consider top num_singular_vals while doing reconstruction

    Eigen::MatrixXd S = SVec.asDiagonal();
    int k = std::min(num_singular_vals, (int) SVec.size());
    cout << "k: " << k << endl;

    Eigen::MatrixXd recon = U.block(0,0,U.rows(),k)\
                            *S.block(0,0,k,k)\
                            *V.block(0,0,V.rows(),k).transpose();

    vector<vector<double>> recon_prob_mat = ConvertEigenToVec(recon, -1);

    if (VERBOSE) {
      bool close = (recon - prob_eigen).isMuchSmallerThan(0.01);
      cout << "reconstruction allclose: " << close << endl;
      cout << "orig prob.size " << prob_matrix.size() << endl;
      cout << "orig prob[0].size " << prob_matrix[0].size() << endl;
      cout << "recon prob.size " << recon_prob_mat.size() << endl;
      cout << "recon prob[0].size " << recon_prob_mat[0].size() << endl;
      cout << "prob_eigen.rows " << prob_eigen.rows() << endl;
      cout << "prob_eigen.cols " << prob_eigen.cols() << endl;

      // just to check since easier to do linear algebra comparisons after
      // conversion to Eigen
      Eigen::MatrixXd prob_eigen2 = ConvertToEigenMatrix(recon_prob_mat);
      bool close2 = (prob_eigen - prob_eigen2).isMuchSmallerThan(0.01);

      cout << "prob_eigen2.rows " << prob_eigen2.rows() << endl;
      cout << "prob_eigen2.cols " << prob_eigen2.cols() << endl;
      cout << "reconstruction to c++, close: " << close2 << endl;
    }

    //this->prob_matrix = recon_prob_mat;
  }

	double cal_mutual_info()
	{
		double mutal_info=0.0;
		double epsilon=0.000001/(1.000*col_num*row_num);
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
    if (!VERBOSE) return;
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
  bool use_svd;
  int num_singular_vals;

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
        temp_edge.init(node_list[i].alphabet_size,data_matrix[i],node_list[j].alphabet_size,data_matrix[j],count_column, use_svd, num_singular_vals);
				edge_matrix[i].push_back(temp_edge);
			}
		}
	}

  /* TODO: describe.
   */
  void update_joint_svd_all()
  {
		for (int i = 0; i < edge_matrix.size(); i++)
		{
			for(int j = 0; j < edge_matrix[i].size(); j++)
			{
        int node2_idx = j+i+1;
        // sanity checks
        Nodes *node1 = &node_list[i];
        Nodes *node2 = &node_list[node2_idx];
        Edges *edge = &edge_matrix[i][j];

        _ASSERT(node1->prob_list.size(), node1->alphabet_size);
        _ASSERT(node2->prob_list.size(), node2->alphabet_size);
        _ASSERT(edge->prob_matrix.size() * edge->prob_matrix[0].size(),
            node1->alphabet_size*node2->alphabet_size);
        _ASSERT(edge->prob_matrix.size(),
            node1->alphabet_size);
        _ASSERT(edge->prob_matrix[0].size(),
            node2->alphabet_size);

        // now we are ready to update the edge
        edge->update_joint_svd_edge(node1->prob_list, node2->prob_list,
            num_singular_vals);
			}
		}
  }

	void init(vector<vector<int> > &data_matrix, vector<int> &count_column,
      bool use_svd, int num_singular_vals)
	{
    if (VERBOSE) {
      cout << "init!" << endl;
      cout << "use svd: " << use_svd << endl;
      cout << "num svs: " << num_singular_vals	 << endl;
    }
    this->use_svd = use_svd;
    this->num_singular_vals = num_singular_vals;

		fill_data(data_matrix,count_column);

    // at this point, the edge matrix would have been initialized. So we can
    // update the joint distribution matrices stored in each edge
    if (use_svd) update_joint_svd_all();
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
        if (VERBOSE) {
          cout<< temp.val <<" : mutual info "<<i<<" "<<j<<endl;
        }
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
          double joint_prob;

          if (child_node < curr_node) {
            joint_prob = temp_edge->prob_matrix[child_val][par_val];
          } else {
            joint_prob = temp_edge->prob_matrix[par_val][child_val];
          }

          if (use_svd && SVD_VERSION == 2) {
            // In this case, we were just storing the difference from the
            // independence assumption. So we just add the independence back
            // in.
            double prob_child = node_list[child_node].prob_list[child_val];
            joint_prob += (prob_child * prob_par);
          }

					ans+= (joint_prob * it_kid->second) / prob_par;
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
    if (!VERBOSE) return;
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
        //edge_matrix[i][j].print();
			}

		}
	}
};

Graphical_Model pgm;

void init(vector<vector<int> > &data_matrix,vector<int> &count_column,
    bool use_svd, int num_singular_vals)
{
	pgm.init(data_matrix,count_column, use_svd, num_singular_vals);
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

extern "C" void py_init(int *data, int row_sz, int col_sz,int *count_ptr,int dim_col, bool use_svd, int num_singular_vals)
{
  vector<vector<int> > data_matrix(col_sz);

  for(int i=0;i<row_sz;i++)
  {
  	for(int j=0;j<col_sz;j++)
  	{
  		data_matrix[j].push_back(*data);
  		data++;
  	}
  }

  vector<int> count_column;

  for(int i=0;i<dim_col;i++)
  {
  	count_column.push_back(*count_ptr);
  	count_ptr++;
  }

  init(data_matrix,count_column, use_svd, num_singular_vals);
  return ;
}

extern "C" void py_train()
{
  train();
  pgm.print();
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
  //cout << "app: " << approx << endl;
  //cout << "frac: " << frac << endl;
  bool app=false;
  if(approx!=0)
  {
    app=true;
  }

  double ans = eval(filter,app,frac);
  return ans;
}

int main(int argc, char *argv[])
{
	fstream file,file1;
  bool use_svd = false;
  int num_singular_vals = -1;

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

	init(data_vec,count_column, use_svd, num_singular_vals);
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

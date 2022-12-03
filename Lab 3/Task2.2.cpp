#include <iostream>
#include <sys/time.h>
#include <thread>
#include <mutex>
#include <list>

using namespace std;
mutex mutex1;

struct Node {
    int key;
    struct Node *left, *right;
};
 
Node* newNode(int item) {
    Node* temp = new Node;
    temp->key = item;
    temp->left = temp->right = NULL;
    return temp;
}
   
/* Computes the number of nodes in a tree. */
int size(Node* node) {
    if (node == NULL) { return 0; }
    else {
        return(size(node->left) + 1 + size(node->right));
    }
}
 
int maxValue(struct Node* node) {
    if (node == NULL) {
        return INT16_MIN;
    }
    int value = node->key;
    int leftMax = maxValue(node->left);
    int rightMax = maxValue(node->right);
    return max(value, max(leftMax, rightMax));
}

int minValue(struct Node* node) {
    if (node == NULL) {
        return INT16_MAX;
    }
    int value = node->key;
    int leftMax = minValue(node->left);
    int rightMax = minValue(node->right);
    return min(value, min(leftMax, rightMax));
}
 
 
/* Returns true if a binary tree is a binary search tree */
int BST(struct Node* node)
{
    if (node == NULL)
        return 1;
    /* false if the max of the left is > than us */
    if (node->left != NULL
        && maxValue(node->left) > node->key)
        return 0;
    /* false if the min of the right is <= than us */
    if (node->right != NULL
        && minValue(node->right) < node->key)
        return 0;
    /* false if, recursively, the left or right is not a BST
     */
    if (!BST(node->left) || !BST(node->right))
        return 0;
    /* passing all that, it's a BST */
    return 1;
}
 
Node* add(Node* node, int key)
{
    if (node == NULL)
        return newNode(key);
 
    /* Otherwise, recur down the tree */
    if (key < node->key)
        node->left = add(node->left, key);
    else
        node->right = add(node->right, key);

    return node;
}

  
 
Node* remove(Node* root, int k)
{   
    // Base case
    if (root == NULL){
        cout << "NULL\n";
        return root;
    }

    if (root->key > k) {
        root->left = remove(root->left, k);
        return root;
    }
    else if (root->key < k) {
        root->right = remove(root->right, k);
        return root;
    }
 
    if (root->left == NULL) {
        Node* temp = root->right;
        delete root;
        return temp;
    }
    else if (root->right == NULL) {
        Node* temp = root->left;
        delete root;
        return temp;
    }
 
    else {
 
        Node* succParent = root;
 
        Node* succ = root->right;
        while (succ->left != NULL) {
            succParent = succ;
            succ = succ->left;
        }

        if (succParent != root)
            succParent->left = succ->right;
        else
            succParent->right = succ->right;

        root->key = succ->key;
 
        delete succ;
        return root;
    }
}
 
int s;

void workload(Node* root) {
    root = NULL;

    //create tree 
    for(int i=0; i<1000; i++){
        int key = rand() % 16384+1;
        if(i % 100 == 0) { mutex1.lock(); }
        root = add(root, key);
        if(i % 100 == 0) {mutex1.unlock(); }
    }    
    
    // delete root
    for(int i=0; i<100000; i++){
        int key = rand() % 16384+1;
        if(i % 100 == 0) { mutex1.lock(); }
        root = add(root, key);
        root = remove(root, key);
        if(i % 100 == 0) { mutex1.unlock(); }
    }
    
    s += size(root);
 
    if (BST(root)) {
        cout << "\n 1 \n";
    }
    else {
        cout << "\n 0 \n";
    }
    
}

int main() {
    srand(time(NULL));
    Node* root = NULL;

    std::list<std::shared_ptr<std::thread>> threads;

    for(int i = 1; i < 17; i++) {
        std::shared_ptr<std::thread> th = std::make_shared<std::thread>(workload,root);
        threads.push_back(th);
    }    

    struct timeval i_time, f_time;
    long mill, sec, usecond;

    gettimeofday(&i_time, NULL);
 
    for(auto & th : threads) {
        th->join();
    }

    gettimeofday(&f_time, NULL);

    cout << "Size of the tree is " << s << "\n";

    sec = f_time.tv_sec - i_time.tv_sec;
    usecond = f_time.tv_usec - i_time.tv_usec;
    mill = ((sec) * 1000 + usecond/1000.0);

    cout << "Execution Time: " << mill << " milliseconds\n";
    
    threads.clear();
       
    return 0;
}